"""This example runs the 'fd/scrfd' face detector on a provided image,
and then draws the face detector information onto a copy of the image.

Example usage:
python face_detection.py -i /path/to/input_image.png -o /path/to/output_image.png
"""

# Standard imports:
import argparse
from pathlib import Path

# Toolkit import:
import fiqat


def main():
  # Parse command-line arguments:
  parser = argparse.ArgumentParser(prog='Face detection example.')
  parser.add_argument('-i', '--input', type=str, required=True, help='A path to an input face image.')
  parser.add_argument('-o', '--output', type=str, required=True, help='The output image will be saved to this path.')
  args = parser.parse_args()

  input_image_path = Path(args.input)
  output_image_path = Path(args.output)

  # Run the face detector:
  face_detector_output: fiqat.FaceDetectorOutput = fiqat.detect_faces('fd/scrfd', input_image_path)

  # Select a primary face (this face will be colored differently in the example output):
  primary_face_estimate: fiqat.PrimaryFaceEstimate = fiqat.estimate_primary_faces('pfe/sccpfe', face_detector_output)

  # Draw the face detector information:
  output_image = fiqat.draw_face_detector_output(
      input_image=input_image_path,
      face_detector_output=primary_face_estimate.face_detector_output,
      primary_face_index=primary_face_estimate.index,
      draw_face_labels=True,
      draw_confidence=True,
      draw_landmark_labels=True,
      font_scale=0.25,
      # method_registry_entry=fiqat.registry.get_method('fd/scrfd'),  # For detector-specific landmark names.
  )

  # Save the output image:
  fiqat.save_image(output_image_path, output_image)


if __name__ == '__main__':
  main()
