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

