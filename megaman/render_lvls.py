import os

#from mario.level_utils import load_level_from_text
from level_image_gen import LevelImageGen
from tokens import REPLACE_TOKENS
# Renders all level.txt files to png images inside a given folder. Expects ONLY .txt in that folder.
def load_level_from_text(path_to_level_txt, replace_tokens=REPLACE_TOKENS):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level
SPRITE_PATH = os.getcwd() + '\\sprites'

if __name__ == '__main__':
    ImgGen = LevelImageGen(SPRITE_PATH)

    directory = '../input/megaman/trimmed/'
    dir_names = os.listdir(directory)
    dir_names.sort()
    if 'README.txt' in dir_names:  # Ignore readme for default input folder
        dir_names.remove('README.txt')

    directory_gen = directory
    names = dir_names
    names.sort()

    target_dir = '../output/level_images/'  # + curr_gen
    os.makedirs(target_dir, exist_ok=True)

    for i in range(min(10, len(names))):
        lvl = load_level_from_text(os.path.join(directory_gen, names[i]), {})
        if lvl[-1][-1] == '\n':
            lvl[-1] = lvl[-1][0:-1]
        lvl_img = ImgGen.render(lvl)
        lvl_img.save(os.path.join(target_dir, names[i][0:-4] + '.png'), format='png')