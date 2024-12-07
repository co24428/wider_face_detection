import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from time import sleep

original_mat_file = '/Users/yihwan/python_project/side_project/wider_face_detection/raw_datasets/wider_face_split/wider_face_train.mat'

def draw_bbox_using_original_data(mat_path, category, image_num):
    mat = scipy.io.loadmat(mat_path)
    
    # Retrieve data
    event_list = mat['event_list']
    file_list = mat['file_list']
    face_bbx_list = mat['face_bbx_list']
    
    # event_name = event_list[0][0][0]  # event name (first directory) (ex. 0--Parade)
    # file_name = file_list[0][0][0][0]  # file name -> ['0_Parade_marchingband_1_849']
    # bboxes = face_bbx_list[0][0][0]  # bounding boxes (x, y, width, height)
    
    # event_name = event_list[1][0][0]  # 1--Handshaking
    # file_name = file_list[1][0][4][0]  # ['1_Handshaking_Handshaking_1_409']
    # bboxes = face_bbx_list[1][0][4]  # bounding boxes (x, y, width, height)
    
    # event_name = event_list[3][0][0]  # 1--Handshaking
    # file_name = file_list[3][0][10][0]  # ['1_Handshaking_Handshaking_1_409']
    # bboxes = face_bbx_list[3][0][10]  # bounding boxes (x, y, width, height)
    
    event_name = event_list[category][0][0]  # 1--Handshaking
    file_name = file_list[category][0][image_num][0]  # ['1_Handshaking_Handshaking_1_409']
    bboxes = face_bbx_list[category][0][image_num]  # bounding boxes (x, y, width, height)
    
    file_name = file_name[0]
    bboxes = bboxes[0]

    print("Event Name:", event_name)
    print("File Name:", file_name)
    print("Bounding Boxes:", bboxes)  # (x, y, width, height) -> [[449 330 122 149]] 
    
    image_path = f"/Users/yihwan/python_project/side_project/wider_face_detection/data/images/train/{event_name}/{file_name}.jpg"  # 이미지 경로
    image = Image.open(image_path)
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for bbox in bboxes:
        x, y, w, h = bbox  # x, y는 왼쪽 상단 좌표, w는 폭, h는 높이
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    
    return 0

def draw_multiple_images(mat_path, category, num):
    for i in range(num):
        draw_bbox_using_original_data(mat_path, category, i)

draw_multiple_images(original_mat_file, 3, 10)
