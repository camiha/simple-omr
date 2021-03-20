import numpy as np
import pandas as pd
import cv2
import sys

marker_threshold_value = 0.7
bin_threshold_value = 120
questions = 50

# メイン処理
def main():

    column_1 = ['grade','class','number']

    for i in range(1,questions+1):
        column_1.append('Q{}'.format(i))

    df1 = pd.DataFrame(columns = column_1)

    print("sheet count = ", end="")
    sheet_count = int(input())

    marker = cv2.imread('marker/marker.png', 0)

    for sheet_id in range(sheet_count):
        
        omr_raw_student_data = []
        omr_raw_answer_area1 = []
        omr_raw_answer_area2 = []

        # グレースケールで外部画像データ読み取り
        mark_sheet = cv2.imread("scandata/" + str(sheet_id+1) + ".png", 0)
        answer_area = identifyMarkArea(marker,mark_sheet)

        # col,row,margin_top,margin_bottom,img
        resized_answer_area = omrSetUpResize(35,25,1,0,answer_area)

        cv2.imwrite('out/resized_answer_area.png', resized_answer_area)
        
        ## 生徒情報処理
        name_tag = resized_answer_area[700 : 1700, 190 : 690]

        name_tag = cv2.rotate(name_tag, cv2.ROTATE_90_COUNTERCLOCKWISE)
        omr_raw_student_data = setUpStudentData(omr(10,5,name_tag))

        ## 成績情報処理１
        answer_area1 = resized_answer_area[100 : 2600, 1050 : 2050]
        omr_raw_answer_area1 = omr(10,25,answer_area1)

        ## 成績情報処理２
        answer_area2 = resized_answer_area[100 : 2600, 2415 : 3415]
        omr_raw_answer_area2 = omr(10,25,answer_area2)

        omr_raw_answer = omr_raw_answer_area1 + omr_raw_answer_area2

        answer = toWriteData(omr_raw_student_data,omr_raw_answer)

        df2 = pd.DataFrame(data = answer,columns = column_1)
        df1 = df1.append(df2, ignore_index = True)

    df1.to_csv("result.csv")

def identifyMarkArea(marker,mark_sheet):

    mark_area = {}

    # マーカー位置の特定
    marker_location = cv2.matchTemplate(mark_sheet, marker, cv2.TM_CCOEFF_NORMED)
    # 特定用閾値の設定
    marker_location = np.where( marker_location >= marker_threshold_value )

    # マーク位置の切り取り
    mark_area['topX'] = min(marker_location[1])
    mark_area['topY'] = min(marker_location[0])
    mark_area['botX'] = max(marker_location[1])
    mark_area['botY'] = max(marker_location[0])

    answer_area = mark_sheet[mark_area['topY']:mark_area['botY'],mark_area['topX']:mark_area['botX']]

    return answer_area

def omrSetUpResize(col,row,margin_top,margin_bot,img):
    row_total = row + margin_top + margin_bot
    resized_img = cv2.resize(img, (col*100, row_total*100))

    return resized_img

def omr(col,row,img):
    omr_raw = []

    blur_img = cv2.GaussianBlur(img,(25,25),0)
    binarization_img = blur_img.copy()

    binarization_img[blur_img < bin_threshold_value] = 0
    binarization_img[blur_img >= bin_threshold_value] = 255

    reversed_img = 255 - binarization_img


    for row in range(1, row+1):
        current_img = reversed_img [(row-1)*100:row*100,]
        area_sum = []

        for current_col in range(col):
            area_sum.append(np.sum(current_img[:,current_col*100:(current_col+1)*100]))
        
        omr_raw.append(area_sum > np.median(255*300))

    return omr_raw

def setUpStudentData(omr_raw):
    raw_student_data = []
    student_data = [0,0,0]
    
    # omr結果を変換
    for x in range(len(omr_raw)):
        temp = np.where(omr_raw[x]==True)[0]
        raw_student_data.append(int(temp))

    raw_student_data = raw_student_data[::-1]
    # grade
    student_data[0] = int(raw_student_data[0])
    # classNumber
    student_data[1] = int(str(raw_student_data[1]) + str(raw_student_data[2]))
    # studentNumber
    student_data[2] = int(str(raw_student_data[3]) + str(raw_student_data[4]))

    return student_data

def toWriteData(omr_raw_student_data,omr_raw_answer):
    write_data = []

    for i in range(3):
        write_data.append(omr_raw_student_data[i])

    for x in range(len(omr_raw_answer)):
        temp = np.where(omr_raw_answer[x]==True)[0]+1
        if len(temp)>1:
            write_data.append('overlap')
        elif len(temp)==1:
            write_data.append(int(temp))
        else:
            write_data.append('no data')

    write_data = [write_data]

    return write_data

if __name__=='__main__':
    main()
