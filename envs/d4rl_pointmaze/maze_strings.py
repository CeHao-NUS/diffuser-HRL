
M_MAZE_V1= \
        "####################\\"+\
        "#OOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO############OOO#\\"+\
        "#O#OO#####O####OO#O#\\"+\
        "#O##OO####O###OO##O#\\"+\
        "#O###OO###O##OO###O#\\"+\
        "#O####OO##O#OO####O#\\"+\
        "#O#####OO#OOO#####O#\\"+\
        "#O######OOOO######O#\\"+\
        "###OOOOOOOOO#OOOOOO#\\"+\
        "#O######OOOO########\\"+\
        "#O#####OO##OO#####O#\\"+\
        "#O####OO####OO####O#\\"+\
        "#O###OO##O###OO###O#\\"+\
        "#O##OO###O####OO##O#\\"+\
        "#O#OO####O#####OO#O#\\"+\
        "#OOO#####O######OOO#\\"+\
        "#OOOOOOOOOOOOOO#OOO#\\"+\
        "####################"

M_MAZE2_V1 = \
"#################\\"+\
"#OOOOO###########\\"+\
"#OOOOOO##########\\"+\
"#####OOOOOO######\\"+\
"######OOOOOO#####\\"+\
"#OOO######OOO####\\"+\
"#OOOO######OOO###\\"+\
"###OOO######OOO##\\"+\
"####OOO######OOO#\\"+\
"#####OO#######OO#\\"+\
"####OOO#######OO#\\"+\
"###OOO########OO#\\"+\
"##OOO#######OOO##\\"+\
"#OOO######OOOO###\\"+\
"#OO######OOO#####\\"+\
"#OOOOOOOOOO######\\"+\
"#OOOOOOOOO#######\\"+\
"#################"



test_maze = \
"######\\"+\
"#OOOO#\\"+\
"####O#\\"+\
"#OOOO#\\"+\
"#OO#O#\\"+\
"#O####\\"+\
"#OOOO#\\"+\
"######"

test_maze_big = \
"#####################\\"+\
"#O#O#OOOOO#OOOOOOOOO#\\"+\
"#O#O#OOOOO#OOOOOOOOO#\\"+\
"#O#OOO#OOO#######OOO#\\"+\
"#O#OOO#OOOOOOOOOOOOO#\\"+\
"#O#####OOOOOOOOOOOOO#\\"+\
"#OOOOOOOOO#OO########\\"+\
"#OO######O#OO#OOOOOO#\\"+\
"#OOOOOOO#O#OO#OOOOOO#\\"+\
"#OOOOOOO#O##O#OO#OOO#\\"+\
"#OOOOOOO#OO#O#OO#OOO#\\"+\
"#OOO#OOO#OO#O#OO#OOO#\\"+\
"#OOO#OOOOOOOO#OO#OOO#\\"+\
"#OOO#OOOOOOOO#####OO#\\"+\
"#OOO#OOOOO#OOOOOOOOO#\\"+\
"###O#######OOOOOOOOO#\\"+\
"#OOOOO#OOO#OO#####OO#\\"+\
"#OOOOOOOOO#OO#OOOOOO#\\"+\
"#OOOOO#OOOOOO#OOOOOO#\\"+\
"#OOOOO#OOOOOO#OOOOOO#\\"+\
"#####################"



maze_name_space= {
    'm_maze1' : M_MAZE_V1,
    'm_maze2' : M_MAZE2_V1,
    'test_maze' : test_maze,
    'test_mazebig' : test_maze_big
}

LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"

HARD_EXP_MAZE = \
        "#####################\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOOOOOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOG#\\"+\
        "#####################"


HARD_EXP_MAZE_V2 = \
        "#####################\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOOOOOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#################OG##\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOOOOOOO#OOOO#\\"+\
        "#OOOO#OOO###OOO#OOOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOOOOOOO#OOOOOOOOO#\\"+\
        "#OOO###OOO#OOO###OOO#\\"+\
        "#OOOO#OOOO#OOOO#OOOO#\\"+\
        "#####################"