# Code based on https://github.com/alts/karel
#-*- coding: utf-8 -*-
from __future__ import print_function

import re
import numpy as np
from collections import Counter

#from .hero import Hero
from .utils import Tcolors, get_rng

def draw2d(array):
    print("\n".join(["".join(["#" if val > 0 else "." for val in row]) for row in array]))

def border_mask(array, value):
    array[0,:], array[-1,:], array[:,0], array[:,-1] = value, value, value, value

def event_callback(block_name, block_span,  cond_span, cond_value,
        selected_span):
    '''
    block_name: if, ifElse, while, repeat
    block_span: (m, n) where
    - m: index of IF/IFELSE/WHILE/REPEAT
    - n: index of i) e) w) r)
    cond_span: (m, n) where
    - m: index of first token in condition/repetitions (excluding "c(")
    - n: index of last token in condition/repetitions (excluding "c)")
    cond_value: True, False, or number
    selected_span: (m, n) where
      if cond_value is True or number < repetitions
      - m: index of i( w( r(
      - n: index of i) w) r)
      else
      - m: block_span[1] or index of e(
      - n: block_span[1] or index of e)
  '''
    raise NotImplementedError


class KarelRuntime(object):
    HERO_CHARS = u'↑>↓<'
    HERO_COMB_CHARS = u'\u0305\u0355\u0332\u0354'
    WALL_CHAR = u'█'
    OBSTACLE_CHAR = u'░'
    EMPTY_CHAR = u' '
    # (0, 0) is at bottom left corner; (h, w) is at top right
    DIRECTIONS = (
        (1, 0),   # north
        (0, 1),   # east
        (-1, 0),  # south
        (0,  -1),  # west
    )

    def __init__(self,  action_callback=None):
        if action_callback is None:
            self.action_callback = lambda *args: None
        else:
            self.action_callback = action_callback

        if event_callback is None:
            self.event_callback = lambda *args: None
        else:
            self.event_callback = event_callback

        # Indiator array of size 15 x height x width (4 <= height, width <= 18)
        # 1st axis:
        #   0: Hero facing North
        #   1: Hero facing East
        #   2: Hero facing South
        #   3: Hero facing West
        #   4: Internal walls
        #   5: Surrounding walls
        #   6: 1 marker
        #   7: 2 markers
        #   8: 3 markers
        #   9: 4 markers
        #   10: 5 markers
        #   11: 6 markers
        #   12: 7 markers
        #   13: 8 markers
        #   14: 9 markers
        # Borders of array have the surrounding walls bit set.
        self.world = None
        self.hero_pos = None
        self.hero_dir = None
        self.nonwalls = None

    def init_randomly(self, world_size, max_marker_in_cell,  wall_ratio,
        marker_ratio, rng=None):
        rng = get_rng(rng)
        height, width = world_size

        if height < 2 or width < 2:
            raise Exception(" [!] `height` and `width` should be at least 2")
        elif height > 16 or width > 16:
            raise Exception(" [!] `height` and `width` should be at most 16")

        # blank world
        self.world = np.zeros((15, height + 2, width + 2), dtype=np.bool)

        # internal walls
        wall_array = rng.rand(height + 2, width + 2)
        self.world[4][wall_array < wall_ratio] = 1
        # external wall
        border_mask(self.world[5], 1)

        # hero
        x = rng.randint(1, width)
        y = rng.randint(1, height)
        self.hero_pos = np.array([y, x])
        self.hero_dir = rng.randint(4)
        self.world[self.hero_dir, y, x] = 1

        # markers
        marker_array = rng.rand(height + 2, width + 2)
        marker_array = (wall_array >= wall_ratio) & (marker_array < marker_ratio)
        border_mask(marker_array, False)

        # TODO Allow more than one marker at a given location
        self.world[6][marker_array > 0] = 1

        # Pad world to 18x18
        #self.world = np.pad(self.world, ((0, 0), (0, 18 - self.world.shape[0]),
        #                                 (0, 18 - self.world.shape[1])),
        #                    'constant', 0)
        self.compute_nonwalls()


    def draw(self, prefix="", skip_number=False, with_color=False, no_print=False):
        canvas = np.full(self.world.shape[1:], self.EMPTY_CHAR, dtype='U2')
        canvas[self.world[4]] = self.OBSTACLE_CHAR
        canvas[self.world[5]] = self.WALL_CHAR
        for count, i in enumerate(range(6, 15)):
            canvas[self.world[i]] = str(count + 1)
        if canvas[tuple(self.hero_pos)] == self.EMPTY_CHAR:
            canvas[tuple(self.hero_pos)] = self.hero_char()
        else:
            canvas[tuple(self.hero_pos)] += self.HERO_COMB_CHARS[self.hero_dir]

        texts = []
        for i in range(self.world.shape[1] - 1, -1, -1):
            text = ''.join(canvas[i])
            if not no_print:
                print(text)
            texts.append(text)

        if no_print:
            return texts

    @property
    def state(self):
        return self.world

    def init_from_array(self, state):
        ys, xs = np.where(state[5])
        height, width = ys.max() + 1, xs.max() + 1
        self.world = state[:, :height, :width]

        pos = list(zip(*np.where(np.any(state[:4], axis=0))))
        if len(pos) > 1:
            raise ValueError('Invalid state: too many hero positions')
        self.hero_pos = np.array(pos[0])

        direction, = np.where(np.any(state[:4], axis=(1, 2)))
        if len(direction) > 1:
            raise ValueError('Invalid state: too many hero directions')
        self.hero_dir = direction[0]
        self.compute_nonwalls()

    def compute_nonwalls(self):
        self.nonwalls = np.logical_not(self.world[4:6].any(axis=0))

    def draw_exception(self, exception):
        pass

    def hero_char(self):
        return self.HERO_CHARS[self.hero_dir]

    def move(self, metadata=None):
        '''Move'''
        if not self.frontIsClear():
            retval = False
        else:
            self.world[self.hero_dir][tuple(self.hero_pos)] = False
            self.hero_pos += self.DIRECTIONS[self.hero_dir]
            self.world[self.hero_dir][tuple(self.hero_pos)] = True
            retval = True

        self.action_callback('move', retval, metadata)
        return retval

    def turn_left(self, metadata=None):
        '''Turn left'''
        self.world[self.hero_dir][tuple(self.hero_pos)] = False
        self.hero_dir -= 1
        self.hero_dir %= 4
        self.world[self.hero_dir][tuple(self.hero_pos)] = True
        self.action_callback('turnLeft', True, metadata)

    def turn_right(self, metadata=None):
        '''Turn right'''
        self.world[self.hero_dir][tuple(self.hero_pos)] = False
        self.hero_dir += 1
        self.hero_dir %= 4
        self.world[self.hero_dir][tuple(self.hero_pos)] = True
        self.action_callback('turnRight', True, metadata)

    def pick_marker(self, metadata=None):
        '''Pick marker'''
        marker_info = self.world[6:15, self.hero_pos[0], self.hero_pos[1]]
        if marker_info[0]:
            marker_info[0] = False
            retval = True
        elif not np.any(marker_info):
            retval = False
        else:
            marker_info[:] = np.roll(marker_info, shift=-1)
            retval = True

        self.action_callback('pickMarker', retval, metadata)
        return retval

    def put_marker(self, metadata=None):
        '''Put marker'''
        marker_info = self.world[6:15, self.hero_pos[0], self.hero_pos[1]]
        if not np.any(marker_info):
            marker_info[0] = True
            retval = True
        elif marker_info[-1]:
            retval = False
        else:
            marker_info[:] = np.roll(marker_info, shift=1)
            retval = True

        self.action_callback('putMarker', retval, metadata)
        return retval

    def front_is_clear(self):
        '''Check front is clear'''
        next_pos = self.hero_pos + self.DIRECTIONS[self.hero_dir]
        return self.nonwalls[next_pos[0], next_pos[1]]

    def left_is_clear(self):
        '''Check left is clear'''
        next_pos = self.hero_pos + self.DIRECTIONS[(self.hero_dir - 1) % 4]
        return self.nonwalls[next_pos[0], next_pos[1]]

    def right_is_clear(self):
        '''Check right is clear'''
        next_pos = self.hero_pos + self.DIRECTIONS[(self.hero_dir + 1) % 4]
        return self.nonwalls[next_pos[0], next_pos[1]]

    def markers_present(self):
        '''Check markers present'''
        return self.world[6:15, self.hero_pos[0], self.hero_pos[1]].any()

    def no_markers_present(self):
        '''Check no markers present'''
        return not self.markers_present()

    @property
    def facing_north(self):
        return self.hero_dir == 0

    @property
    def facing_south(self):
        return self.hero_dir == 2

    @property
    def facing_west(self):
        return self.hero_dir == 3

    @property
    def facing_east(self):
        return self.hero_dir == 1

    @property
    def facing_idx(self):
        return self.hero_dir

    frontIsClear = front_is_clear
    leftIsClear = left_is_clear
    rightIsClear = right_is_clear
    markersPresent = markers_present
    noMarkersPresent = no_markers_present

    turnRight = turn_right
    turnLeft = turn_left
    pickMarker = pick_marker
    putMarker = put_marker
