import os
class Loader:
    def __init__(self):
        self.emotionG = []
        self.gender = []
        self.emotionO = []
        self.path = []
        self.female_ids = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,
            1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,1052,1053,1054,
            1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,
            1082,1084,1089,1091]
        self.temp_dict = {"SAD":"sad", "ANG": "angry", "DIS":"disgust", "FEA":"fear", 
            "HAP":"happy", "NEU":"neutral"}
        
    def get_emotion(self, filename):
        filename = filename.split("_")
        emotionG1 = self.temp_dict[filename[2]]
        if int(filename[0]) in self.female_ids:
            emotionG2 = "_female"
        else:
            emotionG2 = "_male"
        emotionG = emotionG1 + emotionG2
        return (emotionG, emotionG1, emotionG2[1:])

    def load_data(self, CREMA):
        dir_list = os.listdir(CREMA)
        for i in dir_list: 
            self.emotionG.append(self.get_emotion(i)[0])
            self.emotionO.append(self.get_emotion(i)[1])
            self.gender.append(self.get_emotion(i)[2])
            self.path.append(CREMA + i)

    

