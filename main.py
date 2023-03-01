import streamlit as st
from streamlit_option_menu import option_menu
from st_clickable_images import clickable_images
from st_click_detector import click_detector
import cv2
import pickle
import numpy as np
import keras.backend.tensorflow_backend as tb
import glob
# import time

tb._SYMBOLIC_SCOPE.value = True


st.set_page_config(page_title="Batik Detection", page_icon=":camera:", layout="wide")
hide_menu_style = """
        <style>
        MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


button_style = """
        <style>
        .stButton > button {
            color: white;
            background-color: #f44336;
            width: 120px;
            height: 60px;
            text-align:center;
            display: inline-block;
            font-size: 16px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=["Type of Batik","Real Time Detection","About"],
    icons=["easel-fill","camera-fill","exclamation-square-fill"],
    default_index=0,
    orientation="horizontal",
)


@st.cache(allow_output_mutation=True)
def load_images():
    image_files = glob.glob("DatasetsBatikTest/*/*.jpg")
    manuscripts = []
    for image_file in image_files:
        image_file = image_file.replace("\\", "/")
        parts = image_file.split("/")
        if parts[1] not in manuscripts:
            manuscripts.append(parts[1])
    manuscripts.sort()

    return image_files, manuscripts


if selected == "Type of Batik":
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>This is The Introduction About The Type of Batik</h1>", unsafe_allow_html=True)
 
    clicked = clickable_images(
    [
         "https://seragamomahlaweyan.com/wp-content/uploads/2020/12/kawung.jpg",
         "https://seragamomahlaweyan.com/wp-content/uploads/2020/12/megamendung.png",
         "https://www.batikjibb.com/wp-content/uploads/2021/08/08batik.jpg",  
         "https://cdn0-production-images-kly.akamaized.net/TFACTKmWgtYy7WYpYmPey7XCFaA=/469x260/smart/filters:quality(75):strip_icc():format(webp)/kly-media-production/medias/1008004/original/007302900_1443777482-Motif_Sekar_Jagad.jpg",
         "https://dijelasin1-1a502.kxcdn.com/wp-content/uploads/2018/10/2.jpg",
           
    ],
    
    div_style={"display": "flex", "justify-content": "center", },
    img_style={"margin": "5px", "height": "200px", "width":"300px"},
    )
    
    
    if clicked==0:
        with st.container():
        
            st.title("Batik Kawung")
            st.write("Originally from Yogyakarta and the surrounding area.  \nKawung batik is one of the oldest types of batik in Indonesia. Reportedly this motif has two kinds of different meanings. Some people believe that the kawung batik motif is inspired by kawung fruit or palm fruit which is halved.There are some who believe that the kawung motif is inspired by the shape of a type of beetle called kwangwung.The philosophy of the kawung motif is a form of symbolizing the hope that humans will always remember their origins. This motif can also symbolize the four directions and the conscience as the center of controlling the desires that exist in humans so that there is a balance of behavior.")        
            
    elif clicked==1:
        with st.container():
            st.title("Batik Megamendung")
            st.write("Originally from Cirebon, West Java.  \nThe history of Megamendung batik is related to the arrival of the Chinese in Cirebon in the 16th century. At that time, Sunan Gunungjati, who was teaching Islam in Cirebon, married Queen Ong Tien from China.He brought several artworks with cloud motifs from his home country. Clouds are a symbol of the upper world in Taoism and therefore the megamendung motif usually symbolizes the vastness of the world.") 


    elif clicked==2:
        with st.container():
            st.title("Batik Parang")
            st.write("Origin of Kartasura, Central Java.  \nThis batik is one of the oldest Indonesian batik motifs, parang depicts a diagonal  line descending from high to low. The word parang is taken from the word pereng which means slope of a cliff, that's why this batik pattern has a diagonal shape. The parang broken batik motif was created by Panembahan Senopati when he was meditating on the beach. This motif was inspired by the sea waves that never get tired of hitting the coastal rocks.This motif has the meaning of wisdom and the noble character of the character who will win. In addition, this motif also symbolizes power and strength.") 


           
            
    elif clicked==3:
        with st.container():
            st.title("Batik Sekar Jagad")
            st.write("Originally from Solo and Yogyakarta.   \nThis motif contains the meaning of beauty and beauty so that other people who see it will be fascinated. There are also those who think that the Sekar Jagad motif actually comes from the word kar jagad which is taken from the Javanese language (Kar=map; Jagad=world), so this motif also symbolizes diversity throughout the world.The feature of the Sekar Jagad motif which is described as a map is seen in the presence of curved lines resembling the shape of islands next to each other. This motif is unique because it looks as irregular as other batiks that have regular and repetitive patterns. The Sekar jagad batik itself is marked by the presence of various motifs on the islands, such as kawung, truntum, slopes, flora and fauna and others." ) 
            
    elif clicked==4:
        with st.container():
            st.title("Batik Simbut")
            st.write("Originally created by the Baduy tribe, Banten.  \nAs time went on, the Simbut batik motifs increasingly spread throughout Banten. The Simbut batik motifs are in the form of leaves that resemble taro leaves. This motif is the simplest motif, only arranging and tidying up one type of motif. The characteristic of this typical Banten batik motif is its color which tends to be bright, but still not flashy. The lines used in the Simbut batik motif tend to be thick and large in size." ) 
    
    else:
        with st.container():
           st.title("Batik Kawung")
           st.write("Originally from Yogyakarta and the surrounding area.  \nKawung batik is one of the oldest types of batik in Indonesia. Reportedly this motif has two kinds of different meanings. Some people believe that the kawung batik motif is inspired by kawung fruit or palm fruit which is halved.There are some who believe that the kawung motif is inspired by the shape of a type of beetle called kwangwung.The philosophy of the kawung motif is a form of symbolizing the hope that humans will always remember their origins. This motif can also symbolize the four directions and the conscience as the center of controlling the desires that exist in humans so that there is a balance of behavior.")  
            
            
if selected == "Real Time Detection":
    with st.container():
        if st.button("Predict Now !"):
            width = 1280
            height = 720    
            threshold = 0.99
            maxProb = 0
            ResultProb =0 
            
            cam = cv2.VideoCapture(0)
            cam.set(3,width)
            cam.set(4,height)

            pickle_in = open("TheBestModel","rb")
            model = pickle.load(pickle_in)
    
          
            def PreProcessing(img):
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # img = cv2.equalizeHist(img)
                img = img/255 
                return img

            while True:
                success, imgOriginal = cam.read()
                img = np.asarray(imgOriginal)
                img = cv2.resize(img,(128,128))
                img = PreProcessing(img)
                cv2.imshow("Processed Image",img)
                img = img.reshape(1,128,128,3)  
                
                classIndex = int(model.predict_classes(img))
                #print(classIndex)
                predictions = model.predict(img)
                #print(predictions)
                probVal  = np.amax(predictions)
                
                if classIndex == 0:
                    predict = "Kawung"
                elif classIndex == 1:
                    predict = "Megamendung"
                elif classIndex == 2:
                    predict = "Parang"
                elif classIndex == 3:
                    predict = "Sekar Jagad"
                elif classIndex == 4:
                    predict = "Simbut"             
                    
                # if cv2.waitKey(1) & 0xFF == ord(' '):
                ResultProb = probVal 
                ResultPredict = predict
                # elif cv2.waitKey(1) & 0xFF == ord('r'):
                    # ResultProb = 0
                #    cv2.putText(imgOriginal,ResultPredict + " " +str(ResultProb),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1 )
              
                # if ResultProb >threshold: 
                if ResultProb == 1:   
                    cv2.putText(imgOriginal,ResultPredict + " " +str(ResultProb),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1 )
                    # ResetResultProb(int(t))
                elif ResultProb == 0:
                    pass
                else:
                     cv2.putText(imgOriginal,"Not Identified",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1 )
                # if probVal >threshold:
                #     # 
                #         cv2.putText(imgOriginal,predict + " " +str(probVal),(50,50),cv2.FONT_HERSHEY_COMPLEX,
                #             1,(0,0,255),1 )
                
                # if cv2.waitKey(1) & 0xFF == ord(' '):
                #     ResultProb = probVal 
                #     ResultPredict = predict
                #     if ResultProb >threshold:    
                #         cv2.putText(imgOriginal,predict + " " +str(probVal),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1 )
                            
                            
                if probVal > maxProb:
                    maxProb = probVal
                    predictright = predict 
                
                if  cv2.waitKey(1) & 0xFF == ord('q'):    
                    cam.release()
                    break
                
                window_name ="Batik Detection"
                cv2.imshow(window_name, imgOriginal)
                cv2.moveWindow(window_name,5,100)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                

               
            cv2.destroyAllWindows() 
    
 
       
       
if selected== "About":
    image_files, manuscripts = load_images()  
   
    view_manuscripts = st.multiselect("Select Manuscript(s)", manuscripts,default=manuscripts[0])
    n = st.number_input("Select Grid Width", 1, 5, 3)

    view_images = []
    for image_file in image_files:
        if any(manuscript in image_file for manuscript in view_manuscripts):
            view_images.append(image_file)
    groups = []
    for i in range(0, len(view_images), n):
        groups.append(view_images[i:i+n])

    for group in groups:
        cols = st.columns(n)
        for i, image_file in enumerate(group):
            cols[i].image(image_file)
            
     
