# drivecycle-visualization
Drive cycle visualization tool for UWAFT

# Install
Requirements: python3, pip3
- `pip3 install -r requirements.txt`
FFmpeg is requried to output file as MP4.
- Install FFmpeg from: https://www.ffmpeg.org


# Running the script
There are two ways to run the script. The first is using a notebook and the second is using a plain python script.

## Run notebook
- `jupyter notebook`
- Open visualize_drivecycle.ipynb in the browser
- Change the DATA_FILE to the drive cycle file and change FRAMERATE to adjust speed
- Run the entire notebook
- File will output as animation.mp4

## Run script
- Edit visualize_drivecycle.py
- Change the DATA_FILE to the drive cycle file and change FRAMERATE to adjust speed
- `python3 visualize_drivecycle.py`
- File will output as animation.mp4
