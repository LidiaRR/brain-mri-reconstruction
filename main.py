from segmentation import segment
from rendering import rendering
import sys

if len(sys.argv) < 3:
    print("Usage: python main.py <filename> <slice_code>")

else:
    filename = sys.argv[1]
    slice_code = int(sys.argv[2])
    
    segmentation = segment(filename)
    rendering(segmentation, slice=slice_code)