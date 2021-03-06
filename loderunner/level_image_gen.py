import os 

from PIL import Image, ImageOps, ImageEnhance

class LevelImageGen:
    def __init__(self, sprite_path):
        map_names = ['.', 
                    'B', 'b', 
                    '#', 'G', '-', 
                    'E'
                    ]
        sprite_dict = dict()
        for i in range(7):
            imageName = sprite_path + '\\tile'+str(i)+'.png'
            image = Image.open(imageName)
            sprite_dict[map_names[i]] = image
        self.sprite_dict = sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """ Helper to make correct sprites and sprite sizes to draw into the image.
         Some sprites are bigger than one tile and the renderer needs to adjust for them."""

        # Init default size, 8x8 px tokens
        new_left = curr_x * 8 
        new_top = curr_y * 8
        new_right = (curr_x + 1) * 8
        new_bottom = (curr_y + 1) * 8

        actual_sprite = self.sprite_dict[sprite_key]

        return actual_sprite, (new_left, new_top, new_right, new_bottom)

    def render(self, ascii_level):
        """ Renders the ascii level as a PIL Image. Assumes the Background is sky """
        len_level = len(ascii_level[-1])
        height_level = len(ascii_level)

        # Fill base image with empty tiles
        dst = Image.new('RGB', (len_level*8, height_level*8))
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(self.sprite_dict['.'], (x*8, y*8, (x+1)*8, (y+1)*8))

        # Fill with actual tiles
        for y in range(height_level):
            for x in range(len_level):
                curr_sprite = ascii_level[y][x]
                sprite, box = self.prepare_sprite_and_box(ascii_level, curr_sprite, x, y)
                sprite = sprite.convert("RGBA")
                dst.paste(sprite, box, mask=sprite)

        return dst