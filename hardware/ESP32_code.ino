#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <WiFiClientSecure.h>


// --- Wi-Fi and API Configuration ---
const char* ssid = "adhithya";
const char* password = "adhithya365";
const char* api_url = "https://fall-prediction-api.onrender.com/predict/";

String mac_addr_str;

// --- MPU6050 Object ---
Adafruit_MPU6050 mpu;

#define LED_PIN 19

// Function to compute variance
float computeVariance(float data[], int n, float mean) {
  float sumSq = 0;
  for (int i = 0; i < n; i++) {
    float diff = data[i] - mean;
    sumSq += diff * diff;
  }
  return sumSq / n;
}

// --- Send Prediction Request ---
void sendPredictionRequest(
  float max_Ax, float min_Ax, float var_Ax, float mean_Ax,
  float max_Ay, float min_Ay, float var_Ay, float mean_Ay,
  float max_Az, float min_Az, float var_Az, float mean_Az,
  float max_pitch, float min_pitch, float var_pitch, float mean_pitch) {

  WiFiClientSecure client;
  client.setInsecure();   // ⚠ Disable SSL certificate verification
  HTTPClient http;
  http.begin(client, api_url);
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<1024> doc;
  doc["mac_addr"] = mac_addr_str;
  doc["max_Ax"] = max_Ax; doc["min_Ax"] = min_Ax; doc["var_Ax"] = var_Ax; doc["mean_Ax"] = mean_Ax;
  doc["max_Ay"] = max_Ay; doc["min_Ay"] = min_Ay; doc["var_Ay"] = var_Ay; doc["mean_Ay"] = mean_Ay;
  doc["max_Az"] = max_Az; doc["min_Az"] = min_Az; doc["var_Az"] = var_Az; doc["mean_Az"] = mean_Az;
  doc["max_pitch"] = max_pitch; doc["min_pitch"] = min_pitch; doc["var_pitch"] = var_pitch; doc["mean_pitch"] = mean_pitch;

  String json_payload;
  serializeJson(doc, json_payload);

  Serial.println("\nSending JSON payload:");
  Serial.println(json_payload);

  int httpCode = http.POST(json_payload);
  Serial.printf("HTTP POST result code: %d\n", httpCode);

  if (httpCode > 0) {
    String payload = http.getString();
    Serial.println("\nAPI Response:");
    Serial.println(payload);

    StaticJsonDocument<512> response_doc;
    DeserializationError error = deserializeJson(response_doc, payload);

    if (!error) {
      int prediction_value = response_doc["prediction"];
      String prediction_label = response_doc["prediction_label"];

      Serial.print("Prediction: ");
      Serial.println(prediction_label);

      if (prediction_value == 1) {
        Serial.println("Fall detected! Turning LED ON.");
        digitalWrite(LED_PIN, HIGH);
        delay(5000);
      } else {
        Serial.println("No fall detected. LED OFF.");
        digitalWrite(LED_PIN, LOW);
      }
    } else {
      Serial.print("JSON parse error: ");
      Serial.println(error.f_str());
    }
  } else {
    Serial.printf("\nHTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

// --- Setup ---
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  delay(1000);

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi");
  mac_addr_str = WiFi.macAddress();

  // Scan I2C bus
  Serial.println("Scanning I2C bus...");
  Wire.begin();
  byte error, address;
  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("I2C device found at 0x");
      Serial.println(address, HEX);
    }
  }

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip!");
    // while (1) delay(10);
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  delay(100);
}

// --- Loop ---
void loop() {
  const int N = 100; // number of samples
  float Ax[N], Ay[N], Az[N], Pitch[N];

  // Collect 100 samples @ 20ms interval
  for (int i = 0; i < N; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    Ax[i] = a.acceleration.x/9.80655;
    Ay[i] = a.acceleration.y/9.80655;
    Az[i] = a.acceleration.z/9.80655;

    // Compute pitch from accel data (approx)
    Pitch[i] = atan2(Ax[i], sqrt(Ay[i] * Ay[i] + Az[i] * Az[i])) * 180.0 / PI;

    delay(20); // 20ms -> ~50Hz
  }

  // Compute features
  float sumAx = 0, sumAy = 0, sumAz = 0, sumPitch = 0;
  float maxAx = Ax[0], minAx = Ax[0];
  float maxAy = Ay[0], minAy = Ay[0];
  float maxAz = Az[0], minAz = Az[0];
  float maxPitch = Pitch[0], minPitch = Pitch[0];

  for (int i = 0; i < N; i++) {
    sumAx += Ax[i]; if (Ax[i] > maxAx) maxAx = Ax[i]; if (Ax[i] < minAx) minAx = Ax[i];
    sumAy += Ay[i]; if (Ay[i] > maxAy) maxAy = Ay[i]; if (Ay[i] < minAy) minAy = Ay[i];
    sumAz += Az[i]; if (Az[i] > maxAz) maxAz = Az[i]; if (Az[i] < minAz) minAz = Az[i];
    sumPitch += Pitch[i]; if (Pitch[i] > maxPitch) maxPitch = Pitch[i]; if (Pitch[i] < minPitch) minPitch = Pitch[i];
  }

  float meanAx = sumAx / N, meanAy = sumAy / N, meanAz = sumAz / N, meanPitch = sumPitch / N;
  float varAx = computeVariance(Ax, N, meanAx);
  float varAy = computeVariance(Ay, N, meanAy);
  float varAz = computeVariance(Az, N, meanAz);
  float varPitch = computeVariance(Pitch, N, meanPitch);

  // Send features to API
  sendPredictionRequest(
    maxAx, minAx, varAx, meanAx,
    maxAy, minAy, varAy, meanAy,
    maxAz, minAz, varAz, meanAz,
    maxPitch, minPitch, varPitch, meanPitch
  );
}

/**
 * @file ESP32_code.ino
 * @brief Firmware for an ESP32-based fall detection device.
 *
 * This code reads data from an MPU6050 accelerometer and gyroscope,
 * computes statistical features, and sends them to a backend via MQTT.
 * It operates in two modes, selectable by a physical pin:
 * 1. Data Collection Mode: Sends unlabelled data for later analysis and training.
 * 2. Prediction Mode: Sends data for real-time fall detection.
 *
 * The device also subscribes to an MQTT topic to receive fall alerts,
 * triggering an onboard LED.
 */

// --- Required Libraries ---
#include <WiFi.h>
#include <PubSubClient.h> // For MQTT Communication
#include <ArduinoJson.h>  // For creating JSON payloads
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// --- Configuration ---

// 1. Wi-Fi Credentials
const char* ssid = "adhithya";
const char* password = "adhithya365";

// 2. MQTT Broker Configuration
const char* mqtt_broker = "broker.hivemq.com"; // Using public HiveMQ broker
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32-FallDetector-Client"; // Give a unique ID

// 3. MQTT Topics
const char* topic_data_collection = "datacollection";
const char* topic_data_prediction = "dataprediction";
const char* topic_fall_alert = "fallalert";

// 4. Hardware Pin Definitions
#define LED_PIN 19   // Built-in LED on some ESP32 dev boards is GPIO 2
#define MODE_PIN 18  // Pin to switch between modes. Connect to GND to activate prediction mode.

// --- Global Objects ---
WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_MPU6050 mpu;
String mac_addr_str;

// --- Helper Functions ---

/**
 * @brief Computes the variance for an array of floating-point numbers.
 * @param data The array of data points.
 * @param n The number of elements in the array.
 * @param mean The pre-calculated mean of the data.
 * @return The variance of the data.
 */
float computeVariance(const float data[], int n, float mean) {
  float sumSq = 0.0;
  for (int i = 0; i < n; i++) {
    float diff = data[i] - mean;
    sumSq += diff * diff;
  }
  return (n > 0) ? (sumSq / n) : 0.0;
}

/**
 * @brief Callback function for handling incoming MQTT messages.
 * This function is triggered when a message is received on a subscribed topic.
 */
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.println(topic);

  // Check if the message is a fall alert
  if (strcmp(topic, topic_fall_alert) == 0) {
    Serial.println("!!! FALL ALERT RECEIVED !!!");
    digitalWrite(LED_PIN, HIGH); // Turn on the LED to signal an alert
    delay(5000);                 // Keep the LED on for 5 seconds
    digitalWrite(LED_PIN, LOW);  // Turn the LED off
  }
}

/**
 * @brief Connects or reconnects to the MQTT broker.
 * Subscribes to the fall alert topic upon successful connection.
 */
void reconnect_mqtt() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect(mqtt_client_id)) {
      Serial.println("connected");
      // Subscribe to the fall alert topic
      client.subscribe(topic_fall_alert);
      Serial.print("Subscribed to: ");
      Serial.println(topic_fall_alert);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// --- Main Program ---

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for serial connection

  // --- Pin Configuration ---
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  pinMode(MODE_PIN, INPUT_PULLUP); // Use internal pull-up. Pin is HIGH by default.

  // --- Wi-Fi Connection ---
  Serial.print("\nConnecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  mac_addr_str = WiFi.macAddress();
  Serial.print("MAC address: ");
  Serial.println(mac_addr_str);

  // --- MPU6050 Initialization ---
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip. Check connections.");
    while (1) { delay(10); }
  }
  Serial.println("MPU6050 Found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // --- MQTT Client Setup ---
  client.setServer(mqtt_broker, mqtt_port);
  client.setCallback(mqtt_callback);

  Serial.println("\nSetup complete. Entering main loop.");
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  // --- Maintain MQTT Connection ---
  if (!client.connected()) {
    reconnect_mqtt();
  }
  client.loop(); // Allow the MQTT client to process incoming messages

  // --- 1. Data Acquisition ---
  const int NUM_SAMPLES = 100;
  float ax_data[NUM_SAMPLES], ay_data[NUM_SAMPLES], az_data[NUM_SAMPLES];
  float gx_data[NUM_SAMPLES], gy_data[NUM_SAMPLES], gz_data[NUM_SAMPLES];

  Serial.println("\nCollecting 100 sensor samples...");
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    ax_data[i] = a.acceleration.x;
    ay_data[i] = a.acceleration.y;
    az_data[i] = a.acceleration.z;
    gx_data[i] = g.gyro.x;
    gy_data[i] = g.gyro.y;
    gz_data[i] = g.gyro.z;
    delay(20); // Sampling at ~50Hz
  }
  Serial.println("Sample collection complete.");

  // --- 2. Feature Calculation ---
  float sum_ax = 0, max_ax = ax_data[0], min_ax = ax_data[0];
  float sum_ay = 0, max_ay = ay_data[0], min_ay = ay_data[0];
  float sum_az = 0, max_az = az_data[0], min_az = az_data[0];
  float sum_gx = 0, max_gx = gx_data[0], min_gx = gx_data[0];
  float sum_gy = 0, max_gy = gy_data[0], min_gy = gy_data[0];
  float sum_gz = 0, max_gz = gz_data[0], min_gz = gz_data[0];

  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum_ax += ax_data[i]; if (ax_data[i] > max_ax) max_ax = ax_data[i]; if (ax_data[i] < min_ax) min_ax = ax_data[i];
    sum_ay += ay_data[i]; if (ay_data[i] > max_ay) max_ay = ay_data[i]; if (ay_data[i] < min_ay) min_ay = ay_data[i];
    sum_az += az_data[i]; if (az_data[i] > max_az) max_az = az_data[i]; if (az_data[i] < min_az) min_az = az_data[i];
    sum_gx += gx_data[i]; if (gx_data[i] > max_gx) max_gx = gx_data[i]; if (gx_data[i] < min_gx) min_gx = gx_data[i];
    sum_gy += gy_data[i]; if (gy_data[i] > max_gy) max_gy = gy_data[i]; if (gy_data[i] < min_gy) min_gy = gy_data[i];
    sum_gz += gz_data[i]; if (gz_data[i] > max_gz) max_gz = gz_data[i]; if (gz_data[i] < min_gz) min_gz = gz_data[i];
  }

  float mean_ax = sum_ax / NUM_SAMPLES, var_ax = computeVariance(ax_data, NUM_SAMPLES, mean_ax);
  float mean_ay = sum_ay / NUM_SAMPLES, var_ay = computeVariance(ay_data, NUM_SAMPLES, mean_ay);
  float mean_az = sum_az / NUM_SAMPLES, var_az = computeVariance(az_data, NUM_SAMPLES, mean_az);
  float mean_gx = sum_gx / NUM_SAMPLES, var_gx = computeVariance(gx_data, NUM_SAMPLES, mean_gx);
  float mean_gy = sum_gy / NUM_SAMPLES, var_gy = computeVariance(gy_data, NUM_SAMPLES, mean_gy);
  float mean_gz = sum_gz / NUM_SAMPLES, var_gz = computeVariance(gz_data, NUM_SAMPLES, mean_gz);

  // --- 3. JSON Payload Creation ---
  StaticJsonDocument<1024> doc;
  doc["mac_addr"] = mac_addr_str;
  doc["max_Ax"] = max_ax; doc["min_Ax"] = min_ax; doc["var_Ax"] = var_ax; doc["mean_Ax"] = mean_ax;
  doc["max_Ay"] = max_ay; doc["min_Ay"] = min_ay; doc["var_Ay"] = var_ay; doc["mean_Ay"] = mean_ay;
  doc["max_Az"] = max_az; doc["min_Az"] = min_az; doc["var_Az"] = var_az; doc["mean_Az"] = mean_az;
  doc["max_Gx"] = max_gx; doc["min_Gx"] = min_gx; doc["var_Gx"] = var_gx; doc["mean_Gx"] = mean_gx;
  doc["max_Gy"] = max_gy; doc["min_Gy"] = min_gy; doc["var_Gy"] = var_gy; doc["mean_Gy"] = mean_gy;
  doc["max_Gz"] = max_gz; doc["min_Gz"] = min_gz; doc["var_Gz"] = var_gz; doc["mean_Gz"] = mean_gz;

  char json_buffer[1024];
  serializeJson(doc, json_buffer);

  // --- 4. Mode Selection and Publishing ---
  int mode = digitalRead(MODE_PIN);
  const char* target_topic;
  
  if (mode == HIGH) {
    // Pin is not connected to ground -> Data Collection Mode
    target_topic = topic_data_collection;
  } else {
    // Pin is connected to ground -> Prediction Mode
    target_topic = topic_data_prediction;
  }

  Serial.print("Publishing to topic: ");
  Serial.println(target_topic);
  client.publish(target_topic, json_buffer);

  // --- 5. Wait for the next cycle ---
  delay(2000); // Wait 2 seconds before collecting the next batch of data.
}
