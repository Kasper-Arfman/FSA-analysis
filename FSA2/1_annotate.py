import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import Slider
from datetime import datetime
from pyjacket import filetools
from tools.iter_config import iter_measurements
from _script_params import *

mpl.rcParams['keymap.zoom'] = ['a']
mpl.rcParams['keymap.pan'] = []          # Arrow keys by default
mpl.rcParams['keymap.back'] = []
mpl.rcParams['keymap.forward'] = []

AGE = None  # Annotate everything
# AGE = datetime(2025, 5, 14)  # Re-annotate anything older than this date
fm = filetools.FileManager(DATA_ROOT, ANALYSIS_MANUAL)

FULL_SCREEN = False

class AlexAnnotator:
    """ Select ROIs in a two-channel alex image.
    
    Controls:
     - click:       place ROI at mouse position
     - arrow keys:  refine ROI position
     - BACKSPACE:   remove ROI
     - , (<):       previous frame
     - . (>):       next frame
     - 1, 2, ...:   Toggle channel visibility
     - z:           previous ROI
     - x, ENTER:    next ROI
     - ESC:         finish and save output
    
    Inputs:
     - filemanager: filetools.FileManager object
     - filename: MMStack folder containing .ome.tif files
     
     Outputs:
      - {fm.dst_root}/{fm.rel_path}/{filename}/positions.csv
    """
    rect: list[Rectangle]
    text: list[Text]
    
    BRIGHTNESS_SCALING = [60, 99.9]  # Intensity percentiles
    CMAP = [
        LinearSegmentedColormap.from_list('0', [(0, 0, 0), (1, 0, 1)], N=256),
        LinearSegmentedColormap.from_list('1', [(0, 0, 0), (0, 1, 0)], N=256),
    ]
    MAX_FRAMES = 8
    POSITIONS_FILE = 'positions.csv'
    RECT_SIZE = (15, 15)
    ROI_SPEED = 3
    CONFIG_FILE = 'config.json'
    GRIDLINE_SPACING = 100
    MAX_ANGLE = 40
    
    """ === Initialization === """

    def __init__(self, fm: filetools.FileManager, file_name: str, n_channels=None):
        self.fm = fm
        self.file_name = file_name
        self.n_channels = n_channels  # Hardcoded for now.

        # == Read image data
        self.data = self.read_data()
        self.n_frames = self.data.shape[0]
        delta = max(self.n_frames // self.MAX_FRAMES, 1)
        self.frames = [delta*i for i in range(self.MAX_FRAMES)]
        print(f' ==> {self.data.shape = }')
        
        # == Read positions csv
        self.pos = self.read_positions()

        # == Read config
        self.json = self.read_json()
        self.dna_channel = DNA_CH_DEFAULT

        # == Initialize state variables
        self.i_pos = 0  # current position
        self.set_frame(0)
        self.rect = []  # ROI rectangles for drawing
        self.text = []  # ROI text for drawing
        self.visible = [1, 1]
        self.angle = self.json.get('angle', 0)
        
        # == Image Drawing
        self.fig, self.ax = self.init_screen()
        # self.angle_slider = self.add_slider()
        self.toolbar = plt.get_current_fig_manager().toolbar
        self.toolbar.keymap_zoom = ['c']  # Change zoom key to 'c'
        self.draw()
        plt.show()

    def read_data(self):
        """Get data as a lazy image handle"""
        return self.fm.read_img(self.file_name, lazy=True, unzip=self.n_channels)

    def read_positions(self):
        """Read positions from file if it exists"""
        positions_file = self.fm.dst_path(self.POSITIONS_FILE, self.file_name)
        if os.path.exists(positions_file): 
            print('\nReading positions from file:') 
            df = self.fm.read_csv(self.POSITIONS_FILE, folder=self.file_name, dst=True, autodetect=True)   
            print(df)   
        else:
            print('No positions.csv found')
            df = pd.DataFrame()   
        return df

    def read_json(self):
        if not self.fm.exists(self.CONFIG_FILE, self.file_name, dst=True):
            return dict()

        return fm.read_json(self.CONFIG_FILE, self.file_name, dst=True)

    def init_screen(self):
        # create a screen
        fig, ax = plt.subplots(num=self.file_name, figsize=(8, 8))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
        fig.patch.set_facecolor('grey')
        ax.set_axis_off()
        ax.set_facecolor('black')

        # Set axis limits
        H, W = self.frame_data.shape[:2]
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        if FULL_SCREEN:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        
        # Link user-events to callback functions
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Add slider
        slider_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(slider_ax, 'Angle', -self.MAX_ANGLE, self.MAX_ANGLE, valinit=self.angle)
        self.slider.on_changed(self.on_slider)

        return fig, ax   

    """ === Drawing === """

    def draw(self):
        # Wipe data, remembering relevant details
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()

        self.ax.set_title(f'Frame {self.frames[self.i_frame]} - ROI {self.i_pos}/{len(self.pos)-1}', pad=0)
        self.draw_image()
        self.draw_rectangles()
        self.draw_gridlines()

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        return
                    
    def draw_image(self):
        gamma = 0.5  # see dim regions better

        # DNA
        i = self.dna_channel
        if self.visible[i]:
            channel = self.frame_data[..., i]
            self.ax.imshow(channel,
                        cmap=self.CMAP[1],
                        alpha=1,
                        norm=PowerNorm(gamma=gamma, vmin=self.vmins[i], vmax=self.vmaxs[i]))

        # Protein
        i = 1 - self.dna_channel
        if self.visible[i]:
            channel = self.frame_data[..., i]
            self.ax.imshow(channel,
                        cmap=self.CMAP[0],
                        alpha=1/2 if self.visible[self.dna_channel] else 1,
                        norm=PowerNorm(gamma=gamma, vmin=self.vmins[i], vmax=self.vmaxs[i]))
        return

    def draw_rectangles(self, ax: Axes=None):
        """Draw rectangles and update text for current positions."""
        ax = ax if ax is not None else self.ax
        
        # Remove old rectangles
        for rect in self.rect:  rect.remove()
        self.rect = []

        # Remove old text
        for text in self.text:  text.remove()
        self.text = [] 
        
        for i, pos in enumerate(self.pos.itertuples()):
            x, y = pos.x, pos.y
            dx, dy = self.RECT_SIZE
            color = 'w' if i == self.i_pos else 'b'
            
            # ROI rectangle
            rect = Rectangle((x - dx, y - dy), 2*dx, 2*dy, 
                linewidth=1, edgecolor=color, facecolor='none')
            self.rect.append(rect)
            ax.add_patch(rect)
            
            # ROI text
            number = ax.text(x, y - dy - 3, f"{i=}", 
                color=color, fontsize=8, ha='center')
            self.text.append(number)
                  
        plt.draw()

    def draw_gridlines(self):
        spacing = self.GRIDLINE_SPACING
        angle_rad = np.radians(self.angle)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)

        H, W = self.frame_data.shape[:2]
        center_x, center_y = W / 2, H / 2
        n_lines = int(2 * max(H, W) / spacing)

        for i in range(-n_lines // 2, n_lines // 2):
            offset = i * spacing
            x0 = center_x + offset * -dy
            y0 = center_y + offset * dx

            x1 = x0 + dx * W
            y1 = y0 + dy * H
            x2 = x0 - dx * W
            y2 = y0 - dy * H

            self.ax.plot([x1, x2], [y1, y2], 'w--', linewidth=0.8)
            self.ax.set_xlim(0, W)
            self.ax.set_ylim(H, 0)  # Invert y-axis to match image coordinates

    """ === User events === """

    def on_click(self, event: MouseEvent):
        """Handle click events."""
        if self.toolbar.mode or event.inaxes != self.ax:
            return
        self.pos.loc[self.i_pos, 'x'] = int(event.xdata)
        self.pos.loc[self.i_pos, 'y'] = int(event.ydata)
        self.draw()
        # self.draw_rectangles()
     
    def on_key(self, event: KeyEvent):
        """Handle key events."""
        k = event.key
        if k in {'left', 'right', 'up', 'down'}:
            axis, direction = {
                'left':  ('x', -1),
                'right': ('x', +1),
                'up':    ('y', -1),
                'down':  ('y', +1),
            }[event.key]
            self.pos.loc[self.i_pos, axis] += self.ROI_SPEED * direction
            self.draw()
        
        elif k == 'backspace':
            # Delete this ROI
            self.pos.loc[self.i_pos] = np.nan
            self.pos.dropna(inplace=True)
            self.pos.reset_index(drop=True, inplace=True)

            # Go to the previous ROI
            self.on_change_roi(-1) 
        
        elif k == 'z':
            self.on_change_roi(-1) 
        elif k in {'enter', 'x'}:
            self.on_change_roi(+1)
            
        elif k in {'1', '2'}: 
            self.visible[int(k)-1] ^= 1
            self.draw()    
     
        elif k == ',':
            self.on_change_frame(-1) 
        elif k == '.':
            self.on_change_frame(+1) 

        elif k == 'i':
            self.on_flip()

        elif event.key == 'escape':
            self.on_escape()
            plt.close(self.fig)
            return
        
        # self.draw_rectangles(self.ax)
        
    def on_slider(self, angle):
        self.angle = angle
        self.draw()

    def on_change_frame(self, delta: int):
        """Go to a different frame number"""
        self.i_frame = (self.i_frame + delta) % len(self.frames)
        self.frame_data = self.data.get(self.frames[self.i_frame])
        self.draw()

    def on_flip(self):
        self.dna_channel = 1 - self.dna_channel
        self.draw()

    def on_escape(self):
        # Write positions.csv
        if not self.pos.empty:
            self.fm.write_csv(self.POSITIONS_FILE, self.pos, folder=self.file_name)

        # Write config.json
        self.json['angle'] = round(self.angle, 3)
        self.json['dna_ch'] = self.dna_channel
        fm.write_json(self.CONFIG_FILE, self.json, self.file_name)

    def on_change_roi(self, delta: int):
        """Move ROI to a new position"""
        self.i_pos = max(0, min(self.i_pos + delta, len(self.pos)))
        self.draw()

    """ === Getters and Setters === """

    def set_frame(self, i):
        self.i_frame = i
        self.frame_data = self.data.get(self.frames[self.i_frame])

        # Update scaling values
        self.vmins, self.vmaxs = [], []
        for i in range(self.n_channels):
            channel = self.frame_data[..., i]
            vmin, vmax = np.percentile(channel, self.BRIGHTNESS_SCALING)
            self.vmins.append(vmin)
            self.vmaxs.append(vmax)

def main():
    for rel_path, _ in iter_measurements(CONFIG):
        positions_file = fm.dst_path(AlexAnnotator.POSITIONS_FILE, rel_path)

        # Only process the target
        if TARGET:
            if TARGET not in rel_path:  continue

        # Only process missing or old
        elif AGE is not None:
            if filetools.exists(positions_file):
                date_modified = datetime.fromtimestamp(os.path.getmtime(positions_file))
                if date_modified >= AGE:
                    continue

        # Process everything
        else:  pass

        print(f'\n== {rel_path}')
        if not fm.exists('', rel_path):
            raise ValueError('Measurement folder doesnt exist')
        
        AlexAnnotator(fm, rel_path, n_channels=2)

if __name__ == '__main__':
    main()
    print('\nFinished Successfully')
    