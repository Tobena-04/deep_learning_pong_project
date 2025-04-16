import cv2
import numpy as np
import pyautogui
import pytesseract  # For OCR if needed

def preprocess_screen(screen):
    """
    Process a screenshot to extract important game elements
    """
    # Convert to grayscale
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to isolate white elements (ball, paddles, score)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Resize to 80x80 (standard size for many RL implementations)
    resized = cv2.resize(thresh, (80, 80))

    # Normalize to 0-1 range
    normalized = resized / 255.0

    return normalized

def locate_ball(processed_screen):
    """
    Find the ball position in the processed screen
    Returns (x, y) coordinate normalized to 0-1 range
    """
    # Find all white pixels (potential ball locations)
    white_pixels = np.where(processed_screen > 0.5)

    if len(white_pixels[0]) == 0:
        return None  # No ball found

    # Cluster white pixels to find the ball (which should be a small cluster)
    # This is a simplified approach - you might need more sophisticated object detection

    # Calculate the center of the smallest cluster
    # (This assumes the ball is smaller than the paddles)
    # For a simple implementation, we'll find the white pixel furthest from the edges

    # Get coordinates of all white pixels
    coords = np.column_stack((white_pixels[0], white_pixels[1]))

    # Find distances from left and right edges
    distances_from_left = coords[:, 1]
    distances_from_right = 80 - coords[:, 1]

    # Identify pixels that are not near the edges (not paddles)
    middle_pixels = coords[(distances_from_left > 10) & (distances_from_right > 10)]

    if len(middle_pixels) == 0:
        return None  # No ball found

    # Take the mean position as the ball
    ball_y, ball_x = np.mean(middle_pixels, axis=0)

    return ball_x / 80.0, ball_y / 80.0  # Normalize to 0-1

def locate_paddles(processed_screen):
    """
    Find the paddle positions in the processed screen
    Returns (left_y, right_y) paddle centers normalized to 0-1 range
    """
    # Extract the left and right edges of the screen where paddles are usually located
    left_edge = processed_screen[:, :10]
    right_edge = processed_screen[:, -10:]

    # Find white pixels in the left edge (left paddle)
    left_paddle_pixels = np.where(left_edge > 0.5)[0]
    if len(left_paddle_pixels) > 0:
        left_paddle_center = np.mean(left_paddle_pixels)
    else:
        left_paddle_center = 40  # Default to middle if not found

    # Find white pixels in the right edge (right paddle)
    right_paddle_pixels = np.where(right_edge > 0.5)[0]
    if len(right_paddle_pixels) > 0:
        right_paddle_center = np.mean(right_paddle_pixels)
    else:
        right_paddle_center = 40  # Default to middle if not found

    return left_paddle_center / 80.0, right_paddle_center / 80.0  # Normalize to 0-1

def extract_score(screen):
    """
    Extract the score from the screen using simple pixel patterns or OCR
    Returns (left_score, right_score)
    """
    # This is a placeholder implementation
    # For a real implementation, you would use OCR or custom pattern matching

    # Extract the top portion of the screen where scores are displayed
    score_region = screen[20:60, :]

    # For now, we'll return dummy scores
    # In a real implementation, you would use:
    # text = pytesseract.image_to_string(score_region, config='--psm 6')
    # and then parse the text to extract scores

    return 0, 0  # Replace with actual score extraction

def get_game_state(screen=None):
    """
    Extract all relevant game state from a screenshot
    Returns a dictionary with the game state
    """
    if screen is None:
        # Capture screenshot
        screen = np.array(pyautogui.screenshot())

    # Process screen
    processed = preprocess_screen(screen)

    # Extract game elements
    ball_pos = locate_ball(processed)
    left_paddle_pos, right_paddle_pos = locate_paddles(processed)
    left_score, right_score = extract_score(screen)

    # Create state dictionary
    state = {
        'ball_pos': ball_pos,
        'left_paddle_pos': left_paddle_pos,
        'right_paddle_pos': right_paddle_pos,
        'left_score': left_score,
        'right_score': right_score,
        'processed_screen': processed
    }

    return state

# Example usage
if __name__ == "__main__":
    # This would be used for testing
    import matplotlib.pyplot as plt
    import time

    print("Capturing game state in 3 seconds...")
    time.sleep(3)

    # Capture and process a screenshot
    screenshot = np.array(pyautogui.screenshot())
    state = get_game_state(screenshot)

    # Display the processed screen
    plt.figure(figsize=(10, 10))
    plt.imshow(state['processed_screen'], cmap='gray')
    plt.title("Processed Game Screen")

    # Mark the ball and paddle positions
    if state['ball_pos'] is not None:
        ball_x, ball_y = state['ball_pos']
        plt.scatter(ball_x * 80, ball_y * 80, c='r', s=100, label='Ball')

    left_y = state['left_paddle_pos'] * 80
    right_y = state['right_paddle_pos'] * 80
    plt.scatter(5, left_y, c='g', s=100, label='Left Paddle')
    plt.scatter(75, right_y, c='b', s=100, label='Right Paddle')

    plt.legend()
    plt.savefig("game_state_visualization.png")
    print("Game state visualization saved to 'game_state_visualization.png'")

    print(f"Ball position: {state['ball_pos']}")
    print(f"Left paddle position: {state['left_paddle_pos']}")
    print(f"Right paddle position: {state['right_paddle_pos']}")
    print(f"Score: {state['left_score']} - {state['right_score']}")