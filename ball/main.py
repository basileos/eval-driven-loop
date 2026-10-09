import cv2
import numpy as np


def detect_ellipse_occlusion(image_path="frame_02-13_725_original.png"):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Edge detection and contour mapping
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for the contour most likely to be the rim (simplified as the largest valid contour)
    valid_contours = [c for c in contours if len(c) >= 5]
    if not valid_contours:
        return "No valid rim contours found."

    rim_contour = max(valid_contours, key=cv2.contourArea)

    # 2. Fit ellipse to the rim
    ellipse = cv2.fitEllipse(rim_contour)
    (center_x, center_y), (axis_major, axis_minor), angle = ellipse

    # Draw ellipse for visual debugging
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    # 3. Simulate Ball Tracking (Placeholder coordinates)
    # In a full pipeline, you would use cv2.moments() on the ball_mask from the previous script
    # to find the precise centroid (ball_cx, ball_cy) and its radius.
    ball_cx, ball_cy = int(center_x), int(center_y + 10)
    ball_radius = 15

    # 4. Determine spatial relationship
    # The front edge of the rim has a higher Y coordinate (lower on the screen)
    front_edge_y = center_y + (axis_minor / 2)

    # Create a mask for just the front arc
    front_arc_mask = np.zeros_like(gray)
    cv2.ellipse(front_arc_mask, ellipse, 255, 2)
    # Erase the top half of the ellipse to isolate the front arc
    cv2.rectangle(front_arc_mask, (0, 0), (img.shape[1], int(center_y)), 0, -1)

    # Create a mask for the ball
    ball_mask = np.zeros_like(gray)
    cv2.circle(ball_mask, (ball_cx, ball_cy), ball_radius, 255, -1)

    # 5. Check if the ball pixels are missing/occluded exactly where the front arc is
    overlap = cv2.bitwise_and(front_arc_mask, ball_mask)

    if cv2.countNonZero(overlap) > 0:
        return "Geometric intersection: Ball is touching/behind the front rim edge."
    else:
        return "Ball is clear of the front rim edge."

# Execute the function
# print(detect_ellipse_occlusion())

if __name__ == "__main__":
    print(detect_ellipse_occlusion())