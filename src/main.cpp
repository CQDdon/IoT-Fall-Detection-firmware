/**
 * IoT Fall Detection Firmware — ESP32 NodeMCU-32S
 *
 * FreeRTOS dual-core architecture:
 *   Core 0  taskImuReader — Serial2 UART read, WT61PC frame parse, enqueue ImuFrame
 *   Core 1  taskBleTx     — dequeue ImuFrame, pack 61-byte binary, BLE NOTIFY
 *   Core 1  taskLed       — blink LED according to system state
 *
 * Inter-task: FreeRTOS Queue (ImuFrame) + EventGroup (BLE_CONN, IMU_READY)
 *
 * BLE GATT:
 *   Service  SERVICE_UUID
 *     Char   CHAR_IMU_UUID    NOTIFY  — 61-byte binary IMU packet
 *     Char   CHAR_STATUS_UUID READ    — human-readable status string
 *
 * BLE packet layout (61 bytes, all little-endian):
 *   [0-1]   uint8[2]  magic    0xAA 0x55
 *   [2-3]   uint16    seq
 *   [4-15]  float[3]  ax ay az   (m/s²,  ±16 g range)
 *   [16-27] float[3]  gx gy gz   (deg/s, ±2000 dps range)
 *   [28-39] float[3]  roll pitch yaw  (deg)
 *   [40-55] float[4]  q0 q1 q2 q3
 *   [56-59] uint32    esp_ms
 *   [60]    uint8     checksum   XOR of bytes [0..59]
 *
 * Ubuntu Python client (requires: pip install bleak):
 *   import asyncio, struct
 *   from bleak import BleakClient, BleakScanner
 *   CHAR = "12345678-1234-1234-1234-123456789abc"
 *   def cb(_, data):
 *       if len(data) != 61 or data[0] != 0xAA: return
 *       vals = struct.unpack_from('<H13fIB', data, 2)
 *       seq,ax,ay,az,gx,gy,gz,roll,pitch,yaw,q0,q1,q2,q3,ms,csum = vals
 *       print(f"ax={ax:.3f} pitch={pitch:.2f} yaw={yaw:.2f}")
 *   async def main():
 *       dev = await BleakScanner.find_device_by_name("FallDetect-IMU")
 *       async with BleakClient(dev) as c:
 *           await c.start_notify(CHAR, cb)
 *           await asyncio.sleep(3600)
 *   asyncio.run(main())
 */

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/event_groups.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ─── Pin / UART ───────────────────────────────────────────────────────────────
#define IMU_RX_PIN  16
#define IMU_TX_PIN  17
#define LED_PIN      2

static const uint32_t kImuBauds[]  = {9600, 115200};
static const size_t   kImuBaudCnt  = sizeof(kImuBauds) / sizeof(kImuBauds[0]);

// ─── BLE ──────────────────────────────────────────────────────────────────────
#define BLE_DEVICE_NAME  "FallDetect-IMU"
#define SERVICE_UUID     "12345678-1234-1234-1234-123456789012"
#define CHAR_IMU_UUID    "12345678-1234-1234-1234-123456789abc"
#define CHAR_STATUS_UUID "12345678-1234-1234-1234-123456789def"
#define BLE_MTU          512

// ─── FreeRTOS ─────────────────────────────────────────────────────────────────
#define EVT_BLE_CONN   BIT0
#define EVT_IMU_READY  BIT1

#define IMU_Q_LEN 20          // ~1 s buffer at 20 Hz before frames are dropped

// ─── Data types ───────────────────────────────────────────────────────────────
struct ImuFrame {
    float    ax, ay, az;
    float    gx, gy, gz;
    float    roll, pitch, yaw;
    float    q0, q1, q2, q3;
    uint32_t esp_ms;
};

#pragma pack(push, 1)
struct BlePacket {
    uint8_t  magic[2];       // 0xAA 0x55
    uint16_t seq;
    float    ax, ay, az;
    float    gx, gy, gz;
    float    roll, pitch, yaw;
    float    q0, q1, q2, q3;
    uint32_t esp_ms;
    uint8_t  csum;           // XOR bytes[0..59]
};
#pragma pack(pop)

static_assert(sizeof(BlePacket) == 61, "BlePacket size mismatch — check struct layout");

// ─── Global handles ───────────────────────────────────────────────────────────
static QueueHandle_t      g_imuQ     = nullptr;
static EventGroupHandle_t g_events   = nullptr;
static BLEServer*         g_server   = nullptr;
static BLECharacteristic* g_charImu  = nullptr;
static BLECharacteristic* g_charStat = nullptr;

// ─── IMU parser state (accessed only from taskImuReader) ──────────────────────
static struct {
    uint8_t  buf[11];
    uint8_t  idx;
    size_t   baudIdx;
    uint32_t lastValidMs;
    bool     hasAcc;
    bool     hasGyro;
    bool     hasAngle;
    ImuFrame frame;           // accumulates latest decoded values per field
} g_imu;

// ─── WT61PC frame parser ──────────────────────────────────────────────────────
static inline int16_t i16le(const uint8_t *b, int i)
{
    return (int16_t)((uint16_t)b[i] | ((uint16_t)b[i + 1] << 8));
}

static void imuDecodeFrame(const uint8_t *b)
{
    switch (b[1]) {
    case 0x51:
        g_imu.frame.ax = i16le(b, 2) / 32768.0f * 16.0f * 9.80665f;
        g_imu.frame.ay = i16le(b, 4) / 32768.0f * 16.0f * 9.80665f;
        g_imu.frame.az = i16le(b, 6) / 32768.0f * 16.0f * 9.80665f;
        g_imu.hasAcc   = true;
        break;
    case 0x52:
        g_imu.frame.gx = i16le(b, 2) / 32768.0f * 2000.0f;
        g_imu.frame.gy = i16le(b, 4) / 32768.0f * 2000.0f;
        g_imu.frame.gz = i16le(b, 6) / 32768.0f * 2000.0f;
        g_imu.hasGyro  = true;
        break;
    case 0x53:
        g_imu.frame.roll  = i16le(b, 2) / 32768.0f * 180.0f;
        g_imu.frame.pitch = i16le(b, 4) / 32768.0f * 180.0f;
        g_imu.frame.yaw   = i16le(b, 6) / 32768.0f * 180.0f;
        g_imu.hasAngle    = true;
        break;
    case 0x59:
        g_imu.frame.q0 = i16le(b, 2) / 32768.0f;
        g_imu.frame.q1 = i16le(b, 4) / 32768.0f;
        g_imu.frame.q2 = i16le(b, 6) / 32768.0f;
        g_imu.frame.q3 = i16le(b, 8) / 32768.0f;
        break;
    default:
        break;
    }
}

static void imuProcessByte(uint8_t byte)
{
    uint8_t *buf = g_imu.buf;
    uint8_t &idx = g_imu.idx;

    if (idx == 0) {
        if (byte == 0x55) buf[idx++] = byte;
        return;
    }
    buf[idx++] = byte;

    if (idx == 2) {
        if (buf[1] < 0x50 || buf[1] > 0x5A) idx = 0;
        return;
    }
    if (idx < 11) return;

    // Verify checksum, then reset regardless
    uint8_t sum = 0;
    for (int i = 0; i < 10; i++) sum += buf[i];
    uint8_t type = buf[1];
    idx = 0;
    if (sum != buf[10]) return;

    g_imu.lastValidMs = millis();
    imuDecodeFrame(buf);

    // Enqueue a complete snapshot on each angle frame (end of each WT61PC set)
    if (type == 0x53 && g_imu.hasAcc && g_imu.hasGyro) {
        xEventGroupSetBits(g_events, EVT_IMU_READY);
        ImuFrame snap  = g_imu.frame;
        snap.esp_ms    = millis();
        xQueueSend(g_imuQ, &snap, 0);  // drops frame if queue full — non-blocking
    }
}

// ─── BLE server callbacks ─────────────────────────────────────────────────────
class BleCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer *) override {
        xEventGroupSetBits(g_events, EVT_BLE_CONN);
        g_charStat->setValue("connected");
        Serial.println("{\"ble\":\"connected\"}");
    }
    void onDisconnect(BLEServer *srv) override {
        xEventGroupClearBits(g_events, EVT_BLE_CONN);
        g_charStat->setValue("advertising");
        Serial.println("{\"ble\":\"disconnected\"}");
        srv->startAdvertising();
    }
};

// ─── Pack ImuFrame → BlePacket ────────────────────────────────────────────────
static uint16_t g_seq = 0;

static BlePacket makePkt(const ImuFrame &f)
{
    BlePacket p;
    p.magic[0] = 0xAA;
    p.magic[1] = 0x55;
    p.seq      = g_seq++;
    p.ax = f.ax;   p.ay = f.ay;   p.az = f.az;
    p.gx = f.gx;   p.gy = f.gy;   p.gz = f.gz;
    p.roll  = f.roll;
    p.pitch = f.pitch;
    p.yaw   = f.yaw;
    p.q0 = f.q0;   p.q1 = f.q1;
    p.q2 = f.q2;   p.q3 = f.q3;
    p.esp_ms = f.esp_ms;

    uint8_t x = 0;
    const uint8_t *raw = reinterpret_cast<const uint8_t *>(&p);
    for (size_t i = 0; i < 60; i++) x ^= raw[i];
    p.csum = x;
    return p;
}

// ─── Task: IMU Reader — Core 0, priority 3 ───────────────────────────────────
static void taskImuReader(void *)
{
    uint32_t retryMs = millis();

    for (;;) {
        while (Serial2.available())
            imuProcessByte((uint8_t)Serial2.read());

        // Auto-cycle baud rate until first valid frame
        if (!(xEventGroupGetBits(g_events) & EVT_IMU_READY)) {
            uint32_t now = millis();
            if (now - retryMs >= 1500) {
                retryMs = now;
                g_imu.baudIdx = (g_imu.baudIdx + 1) % kImuBaudCnt;
                g_imu.idx     = 0;
                Serial2.end();
                vTaskDelay(pdMS_TO_TICKS(20));
                Serial2.begin(kImuBauds[g_imu.baudIdx], SERIAL_8N1, IMU_RX_PIN, IMU_TX_PIN);
                Serial.printf("{\"imu_baud\":%u}\n", (unsigned)kImuBauds[g_imu.baudIdx]);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// ─── Task: BLE Transmit — Core 1, priority 2 ─────────────────────────────────
static void taskBleTx(void *)
{
    ImuFrame frame    = {};
    uint32_t diagMs   = 0;
    uint32_t pktCount = 0;

    for (;;) {
        if (xQueueReceive(g_imuQ, &frame, pdMS_TO_TICKS(200)) != pdTRUE)
            continue;

        // Drain stale frames while client is disconnected
        if (!(xEventGroupGetBits(g_events) & EVT_BLE_CONN))
            continue;

        BlePacket pkt = makePkt(frame);
        g_charImu->setValue(reinterpret_cast<uint8_t *>(&pkt), sizeof(pkt));
        g_charImu->notify();
        pktCount++;

        // Periodic USB debug line (~every 5 s)
        uint32_t now = millis();
        if (now - diagMs >= 5000) {
            diagMs = now;
            Serial.printf(
                "{\"tx_pkts\":%u,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,"
                "\"roll\":%.2f,\"pitch\":%.2f,\"yaw\":%.2f}\n",
                (unsigned)pktCount,
                frame.ax, frame.ay, frame.az,
                frame.roll, frame.pitch, frame.yaw
            );
        }
    }
}

// ─── Task: Status LED — Core 1, priority 1 ───────────────────────────────────
// LED blink patterns:
//   BLE + IMU : slow pulse   (900 ms on / 900 ms off)  — everything OK
//   BLE only  : medium blink (300 ms on / 300 ms off)  — connected, no IMU
//   IMU only  : fast blink   (100 ms on / 100 ms off)  — IMU OK, advertising
//   none      : double-blink + pause                   — searching
static void taskLed(void *)
{
    for (;;) {
        EventBits_t b   = xEventGroupGetBits(g_events);
        bool        ble = b & EVT_BLE_CONN;
        bool        imu = b & EVT_IMU_READY;

        if (ble && imu) {
            digitalWrite(LED_PIN, HIGH); vTaskDelay(pdMS_TO_TICKS(900));
            digitalWrite(LED_PIN, LOW);  vTaskDelay(pdMS_TO_TICKS(900));
        } else if (ble) {
            digitalWrite(LED_PIN, HIGH); vTaskDelay(pdMS_TO_TICKS(300));
            digitalWrite(LED_PIN, LOW);  vTaskDelay(pdMS_TO_TICKS(300));
        } else if (imu) {
            digitalWrite(LED_PIN, HIGH); vTaskDelay(pdMS_TO_TICKS(100));
            digitalWrite(LED_PIN, LOW);  vTaskDelay(pdMS_TO_TICKS(100));
        } else {
            for (int i = 0; i < 2; i++) {
                digitalWrite(LED_PIN, HIGH); vTaskDelay(pdMS_TO_TICKS(80));
                digitalWrite(LED_PIN, LOW);  vTaskDelay(pdMS_TO_TICKS(80));
            }
            vTaskDelay(pdMS_TO_TICKS(700));
        }
    }
}

// ─── BLE initialisation ───────────────────────────────────────────────────────
static void initBLE()
{
    BLEDevice::init(BLE_DEVICE_NAME);
    BLEDevice::setMTU(BLE_MTU);

    g_server = BLEDevice::createServer();
    g_server->setCallbacks(new BleCallbacks());

    BLEService *svc = g_server->createService(SERVICE_UUID);

    g_charImu = svc->createCharacteristic(
        CHAR_IMU_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    g_charImu->addDescriptor(new BLE2902());

    g_charStat = svc->createCharacteristic(
        CHAR_STATUS_UUID, BLECharacteristic::PROPERTY_READ);
    g_charStat->setValue("advertising");

    svc->start();

    BLEAdvertising *adv = BLEDevice::getAdvertising();
    adv->addServiceUUID(SERVICE_UUID);
    adv->setScanResponse(true);
    adv->setMinPreferred(0x06);  // hint for BLE connection interval
    BLEDevice::startAdvertising();
}

// ─── Arduino entry points ─────────────────────────────────────────────────────
void setup()
{
    Serial.begin(115200);
    delay(300);
    Serial.println("{\"status\":\"boot\"}");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    g_imuQ   = xQueueCreate(IMU_Q_LEN, sizeof(ImuFrame));
    g_events = xEventGroupCreate();
    configASSERT(g_imuQ);
    configASSERT(g_events);

    Serial2.begin(kImuBauds[g_imu.baudIdx], SERIAL_8N1, IMU_RX_PIN, IMU_TX_PIN);
    Serial.printf("{\"imu_baud_start\":%u}\n", (unsigned)kImuBauds[g_imu.baudIdx]);

    initBLE();

    xTaskCreatePinnedToCore(taskImuReader, "imu",   4096, nullptr, 3, nullptr, 0);
    xTaskCreatePinnedToCore(taskBleTx,    "bletx", 8192, nullptr, 2, nullptr, 1);
    xTaskCreatePinnedToCore(taskLed,      "led",   2048, nullptr, 1, nullptr, 1);

    Serial.println("{\"status\":\"running\"}");
}

void loop()
{
    // All logic lives in FreeRTOS tasks.
    vTaskDelay(pdMS_TO_TICKS(5000));
}
