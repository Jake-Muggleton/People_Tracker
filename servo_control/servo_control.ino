#include <Servo.h>

Servo myservo;  
String inByte1 = "";
String inByte2 = "";
String inByte3 = "";
String inByte4 = "";

int pos   = 90; // servo angular position
int buzz  = 0;  // state of buzzer

void setup() {
  Serial.begin(9600);       // set the baud rate
  Serial.println("Ready");  // print "Ready" once
  myservo.attach(9);
  pinMode(8, OUTPUT);
}

void loop() {
  inByte1 = "";
  inByte2 = "";
  inByte3 = "";
  inByte4 = "";

  if(Serial.available()==4){ // only read data if data has been sent
    // data is 4 bytes: pos1, pos2, pos3, buzzerstate
    inByte1 = Serial.read(); // read the incoming data
    inByte2 = Serial.read(); // read the incoming data
    inByte3 = Serial.read(); // read the incoming data
    inByte4 = Serial.read(); // read the incoming data

    pos   = (inByte1.toInt()- '0')*100 + (inByte2.toInt()- '0')*10 +inByte3.toInt() - '0';
    buzz  = inByte4.toInt() - '0';
    
    if(pos >= 0 && pos <= 180)
      myservo.write(pos);
      
    if(buzz == 2)
      digitalWrite(8, HIGH); 
    else
      digitalWrite(8, LOW); 
      
  }

  delay(10); // delay and allow serial buffers to be flushed by NN 
}
