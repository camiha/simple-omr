import numpy as np
import pandas as pd
import cv2
import sys

markerThreshold = 0.7
threshold_value = 120
Questions = 50

# メイン処理
def main():

    columns1 = ['grade','class','number']

    for i in range(1,Questions+1):
        columns1.append('Q{}'.format(i))

    df1 = pd.DataFrame(columns = columns1)

    print("sheet count = ", end="")
    sheetCount = int(input())

    marker = cv2.imread('marker/marker.png', 0)

    for sheetID in range(sheetCount):
        
        studentDataOmrResult = []
        answerArea1OmrResult = []
        answerArea2OmrResult = []

        # グレースケールで外部画像データ読み取り
        markSheet = cv2.imread("scandata/" + str(sheetID+1) + ".png", 0)
        answerArea = identifyMarkArea(marker,markSheet)

        # col,row,marginTop,marginBottom,img
        resizedAnswerArea = omrSetUpResize(35,25,1,0,answerArea)

        cv2.imwrite('out/resizedAnswerArea.png', resizedAnswerArea)
        
        ## 生徒情報処理
        nameTag = resizedAnswerArea[700 : 1700, 190 : 690]

        nameTag = cv2.rotate(nameTag, cv2.ROTATE_90_COUNTERCLOCKWISE)
        studentDataOmrResult = setUpStudentData(omr(10,5,nameTag))

        ## 成績情報処理１
        answerArea1 = resizedAnswerArea[100 : 2600, 1050 : 2050]
        answerArea1OmrResult = omr(10,25,answerArea1)

        ## 成績情報処理２
        answerArea2 = resizedAnswerArea[100 : 2600, 2415 : 3415]
        answerArea2OmrResult = omr(10,25,answerArea2)

        answerOmrResult = answerArea1OmrResult + answerArea2OmrResult

        answer = toWriteData(studentDataOmrResult,answerOmrResult)

        df2 = pd.DataFrame(data = answer,columns = columns1)
        df1 = df1.append(df2, ignore_index = True)

    df1.to_csv("result.csv")

def identifyMarkArea(marker,markSheet):

    markArea = {}

    # マーカー位置の特定
    markerLocation = cv2.matchTemplate(markSheet, marker, cv2.TM_CCOEFF_NORMED)
    # 特定用閾値の設定
    markerLocation = np.where( markerLocation >= markerThreshold )

    # マーク位置の切り取り
    markArea['topX'] = min(markerLocation[1])
    markArea['topY'] = min(markerLocation[0])
    markArea['botX'] = max(markerLocation[1])
    markArea['botY'] = max(markerLocation[0])

    answerArea = markSheet[markArea['topY']:markArea['botY'],markArea['topX']:markArea['botX']]

    return answerArea

def omrSetUpResize(col,row,marginTop,marginBot,img):
    rowTotal = row + marginTop + marginBot
    resizedImg = cv2.resize(img, (col*100, rowTotal*100))

    return resizedImg

def omr(col,row,img):
    omrResult = []

    blurredImg = cv2.GaussianBlur(img,(25,25),0)
    binarizationImg = blurredImg.copy()

    binarizationImg[blurredImg < threshold_value] = 0
    binarizationImg[blurredImg >= threshold_value] = 255

    reversedImg = 255 - binarizationImg


    for row in range(1, row+1):
        currentImg = reversedImg [(row-1)*100:row*100,]
        areaSum = []

        for currentCol in range(col):
            areaSum.append(np.sum(currentImg[:,currentCol*100:(currentCol+1)*100]))
        
        omrResult.append(areaSum > np.median(255*300))

    return omrResult

def setUpStudentData(omrResult):
    studentRawData = []
    studentData = [0,0,0]
    
    # omr結果を変換
    for x in range(len(omrResult)):
        temp = np.where(omrResult[x]==True)[0]
        studentRawData.append(int(temp))

    studentRawData = studentRawData[::-1]
    # grade
    studentData[0] = int(studentRawData[0])
    # classNumber
    studentData[1] = int(str(studentRawData[1]) + str(studentRawData[2]))
    # studentNumber
    studentData[2] = int(str(studentRawData[3]) + str(studentRawData[4]))

    return studentData

def toWriteData(studentDataOmrResult,answerOmrResult):
    convertData = []

    for i in range(3):
        convertData.append(studentDataOmrResult[i])

    for x in range(len(answerOmrResult)):
        temp = np.where(answerOmrResult[x]==True)[0]+1
        if len(temp)>1:
            convertData.append('overlap')
        elif len(temp)==1:
            convertData.append(int(temp))
        else:
            convertData.append('no data')

    convertData = [convertData]

    return convertData

if __name__=='__main__':
    main()
