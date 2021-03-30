import os
import random

def printLevel(level):
    for line in level:
        string = ""
        for char in line:
            string+=char
        print(string)

def converToListOfString(level):
    result = []
    for line in level:
        string = ""
        for char in line:
            string+=char
        result.append(string)
    return result


def convertFileToListOfLists(directory):
    a_file = open(directory, "r")

    list_of_lists = []
    for line in a_file:
        stripped_line = line.strip()
        newList = []
        for char in stripped_line:
            newList.append(char)
        list_of_lists.append(newList)
    a_file.close()
    return list_of_lists
    
def getSegment(level, x, y): #takes in the level as list of list and x-y points of the upper left corner and returns the segment at that area

    result = []
    for newY in range(y, y+14):
        line = []
        for newX in range(x, x+16):
            line.append(level[newY][newX])
        result.append(line)
    return result

def getSegments(level): #takes in a List<List<Integer>> level and returns the distinct segments
    result = []
    stepX = 16
    stepY = 14
    x = 0
    while x <= (len(level[0]) - stepX):
        y = 0
        segment = None
        while y <= (len(level) - stepY):
            if level[y][x] != '@': #grab the entire segment
                segment = getSegment(level, x, y)
                result.append(segment)
            y+=stepY
        x+=stepX

    return result





def convertNullToSegment(level, segment, x, y):
    oX = 0
    oY = 0
    for newY in range(y, y+14):
        oX = 0
        for newX in range(x, x+16):
            # print(oY, oX)
            level[newY][newX] = segment[oY][oX]
            oX+=1
        oY+=1

def replaceNullWithSegmentFromSet(level, listOfSegments):
    # result = []
    stepX = 16
    stepY = 14
    # print((len(level)-stepY), len(level[0]))
    x = 0
    while x <= (len(level[0]) - stepX):
        y = 0
        
        while y <= (len(level) - stepY):
            segment = random.choice(listOfSegments)
            if level[y][x] == '@': #grab the entire segment
                print('Converting from nullspace')
                convertNullToSegment(level, segment, x, y)
                # printLevel(level)
                print('Done!')
                # quit()
            y+=stepY
        x+=stepX
    
    # for segment in result:
    #     printLevel(segment)
    return level




if __name__ == '__main__':
    directory = '../input/megaman/trimmed'
    dir_names = os.listdir(directory)
    if 'README.txt' in dir_names:  # Ignore readme for default input folder
        dir_names.remove('README.txt')

    directory_gen = directory
    names = dir_names
    names.sort()

    target_dir = '../input/megaman/experimental/'  # + curr_gen
    os.makedirs(target_dir, exist_ok=True)

    for i in range(min(1, len(names))):
        level = convertFileToListOfLists(os.path.join(directory_gen, names[i]))
        # printLevel(level)
        segments = getSegments(level)
        replaceNullWithSegmentFromSet(level, segments)
        printLevel(level)

        if level[-1][-1] == '\n':
            level[-1] = level[-1][0:-1]
        levelString = converToListOfString(level)
        with open(target_dir+'test.txt', 'w') as f:
            for item in levelString:
                f.write("%s\n" % item)
        print('completed')
        # print(level[0])
        #lvl = convertFileToListOfLists(os.path.join(directory_gen, names[i]), {})
        #if lvl[-1][-1] == '\n':
        #    lvl[-1] = lvl[-1][0:-1]
        # lvl_img = ImgGen.render(lvl)
        #lvl_img.save(os.path.join(target_dir, names[i][0:-4] + '.txt'), format='txt')
    