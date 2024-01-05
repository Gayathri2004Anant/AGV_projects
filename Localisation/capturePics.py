import cv2
import os

# Create the directory to store the captured images
os.makedirs("calibImgs", exist_ok=True)

# Initialize the webcam
camera = cv2.VideoCapture(0)

# Counter for the captured images
image_counter = 0

while True:
    # Read a frame from the webcam
    ret, frame = camera.read()

    # Display the frame
    cv2.imshow("Capture Images", frame)

    # Wait for the spacebar key
    if cv2.waitKey(1) == ord(" "):
        # Generate a unique filename for each captured image
        filename = f"calibImgs/image{image_counter}.jpg"

        # Save the image
        cv2.imwrite(filename, frame)

        print(f"Image {image_counter} captured and saved.")

        # Increment the image counter
        image_counter += 1

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and close the window
camera.release()
cv2.destroyAllWindows()
