import cv2
import numpy as np

def main():
    """
    Plays four videos in a 2x2 grid with mathematical labels.
    Press 'q' to quit.
    """
    # --- Configuration ---
    video_paths = {
        'top_left': 'No_Force.mp4',
        'top_right': 'Mg_cos(theta)-Mr(thetadot)^2_Force.mp4',
        'bottom_left': 'Mg_Force.mp4',
        'bottom_right': 'Mg_by_cos(theta)_Force.mp4'
    }

    labels = {
        'top_left': 'F = 0 (Free Fall)',
        'top_right': 'F = mg*cos(theta) - m*r*thetadot^2',
        'bottom_left': 'F = mg',
        'bottom_right': 'F = mg / cos(theta)'
    }

    output_filename = 'simulation_grid.mp4'
    # Define a smaller size for each video in the grid
    cell_width, cell_height = 640, 360

    # --- Initialization ---
    caps = {name: cv2.VideoCapture(path) for name, path in video_paths.items()}

    # Check if all videos opened successfully
    for name, cap in caps.items():
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_paths[name]}'")
            return

    # Get FPS from the first video
    fps = caps['top_left'].get(cv2.CAP_PROP_FPS)
    
    # --- Label Rendering ---
    label_height = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    line_type = 2

    # --- Video Writer Initialization ---
    grid_width = cell_width * 2
    grid_height = (cell_height + label_height) * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (grid_width, grid_height))
    print(f"Recording grid video to '{output_filename}'...")

    # --- Main Loop ---
    while True:
        frames = {}
        all_frames_read = True

        # Read a frame from each video
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                # If a video ends, restart it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    all_frames_read = False
                    break
            # Resize the frame to the smaller cell size
            frames[name] = cv2.resize(frame, (cell_width, cell_height))

        if not all_frames_read:
            break

        # Add labels to each frame
        labeled_frames = {}
        for name, frame in frames.items():
            # Create a black canvas for the frame and its label
            labeled_frame = np.zeros((cell_height + label_height, cell_width, 3), dtype=np.uint8)
            # Copy the resized video frame onto the canvas
            labeled_frame[0:cell_height, 0:cell_width] = frame
            # Add the text label below the frame
            cv2.putText(labeled_frame, labels[name], (10, cell_height + 25), font, font_scale, font_color, line_type)
            labeled_frames[name] = labeled_frame

        # Assemble the 2x2 grid
        top_row = np.hstack((labeled_frames['top_left'], labeled_frames['top_right']))
        bottom_row = np.hstack((labeled_frames['bottom_left'], labeled_frames['bottom_right']))
        grid = np.vstack((top_row, bottom_row))

        video_writer.write(grid)

        cv2.imshow('Simulation Grid', grid)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print(f"Finished recording. Video saved as '{output_filename}'.")
    video_writer.release()
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()