import cv2

# Load the image
image = cv2.imread("Data\B\Image_1685262199.5000463.jpg")

# Define the text to display in Malayalam
text = "hi"

# Set the font properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)  # BGR color format
thickness = 2

# Display the Malayalam text on the image
cv2.putText(image, text, (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)

# Show the image
cv2.imshow("Malayalam Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print ("വ")