from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
import streamlit as st
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
BASE_DIR = "/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/model_pkl"


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  image=imread(uploaded_file)
  image_resized = resize(image, (1572, 2213), anti_aliasing=True)
  image_gray = color.rgb2gray(image)
  image_gray=resize(image_gray,(1572,2213))
  """#### **UPLOADED ECG IMAGE**"""
  
  #checkign if we parse the user image and similar to our format
  image1 = imread('/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/ECG_IMAGES_DATASET/ECG Images of Patient that have History of MI (172x12=2064)/PMI(1).jpg')
  image1 = color.rgb2gray(image1)
  image1=resize(image1,(1572,2213))

  image2 = imread('/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/ECG_IMAGES_DATASET/ECG Images of Patient that have abnormal heartbeat (233x12=2796)/HB(6).jpg')
  image2 = color.rgb2gray(image2)
  image2=resize(image2,(1572,2213))

  image3 = imread('/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/ECG_IMAGES_DATASET/Normal Person ECG Images (284x12=3408)/Normal(1).jpg')
  image3 = color.rgb2gray(image3)
  image3=resize(image3,(1572,2213))

  image4 = imread('/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/ECG_IMAGES_DATASET/ECG Images of Myocardial Infarction Patients (240x12=2880)/MI(1).jpg')
  image4 = color.rgb2gray(image4)
  image4=resize(image4,(1572,2213))

  similarity_score = max(
      structural_similarity(image_gray, image1, data_range=1.0),
      structural_similarity(image_gray, image2, data_range=1.0),
      structural_similarity(image_gray, image3, data_range=1.0),
      structural_similarity(image_gray, image4, data_range=1.0)
  )


  if similarity_score > 0.50:
    st.image(image)
    """#### **GRAY SCALE IMAGE**"""
    my_expander = st.expander(label='Gray SCALE IMAGE')
    with my_expander: 
      st.image(image_gray)
    """#### **DIVIDING LEADS**"""
    #dividing the ECG leads from 1-13 from the above image
    Lead_1 = image_resized[300:600, 150:643]
    Lead_2 = image_resized[300:600, 646:1135]
    Lead_3 = image_resized[300:600, 1140:1625]
    Lead_4 = image_resized[300:600, 1630:2125]
    Lead_5 = image_resized[600:900, 150:643]
    Lead_6 = image_resized[600:900, 646:1135]
    Lead_7 = image_resized[600:900, 1140:1625]
    Lead_8 = image_resized[600:900, 1630:2125]
    Lead_9 = image_resized[900:1200, 150:643]
    Lead_10 = image_resized[900:1200, 646:1135]
    Lead_11 = image_resized[900:1200, 1140:1625]
    Lead_12 = image_resized[900:1200, 1630:2125]
    Lead_13 = image_resized[1250:1480, 150:2125]

    Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
    #plotting lead 1-12
    fig , ax = plt.subplots(4,3)
    fig.set_size_inches(10, 10)
    x_counter=0
    y_counter=0

    for x,y in enumerate(Leads[:len(Leads)-1]):
      if (x+1)%3==0:
        ax[x_counter][y_counter].imshow(y)
        ax[x_counter][y_counter].axis('off')
        ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
        x_counter+=1
        y_counter=0
      else:
        ax[x_counter][y_counter].imshow(y)
        ax[x_counter][y_counter].axis('off')
        ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
        y_counter+=1
    
    fig.savefig('Leads_1-12_figure.png')
    fig1 , ax1 = plt.subplots()
    fig1.set_size_inches(10, 10)
    ax1.imshow(Lead_13)
    ax1.set_title("Leads 13")
    ax1.axis('off')
    fig1.savefig('Long_Lead_13_figure.png')
    my_expander1 = st.expander(label='DIVIDING LEAD')
    with my_expander1:
      st.image('Leads_1-12_figure.png')
      st.image('Long_Lead_13_figure.png')

    """#### **PREPROCESSED LEADS**"""
    fig2 , ax2 = plt.subplots(4,3)
    fig2.set_size_inches(10, 10)
    #setting counter for plotting based on value
    x_counter=0
    y_counter=0

    for x,y in enumerate(Leads[:len(Leads)-1]):
      #converting to gray scale
      grayscale = color.rgb2gray(y)
      #smoothing image
      blurred_image = gaussian(grayscale, sigma=0.9)
      #thresholding to distinguish foreground and background
      #using otsu thresholding for getting threshold value
      global_thresh = threshold_otsu(blurred_image)

      #creating binary image based on threshold
      binary_global = blurred_image < global_thresh
      #resize image
      binary_global = resize(binary_global, (300, 450))
      if (x+1)%3==0:
        ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
        ax2[x_counter][y_counter].axis('off')
        ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
        x_counter+=1
        y_counter=0
      else:
        ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
        ax2[x_counter][y_counter].axis('off')
        ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
        y_counter+=1
    fig2.savefig('Preprossed_Leads_1-12_figure.png')
    
    #plotting lead 13
    fig3 , ax3 = plt.subplots()
    fig3.set_size_inches(10, 10)
    #converting to gray scale
    grayscale = color.rgb2gray(Lead_13)
    #smoothing image
    blurred_image = gaussian(grayscale, sigma=0.7)
    #thresholding to distinguish foreground and background
    #using otsu thresholding for getting threshold value
    global_thresh = threshold_otsu(blurred_image)
    print(global_thresh)
    #creating binary image based on threshold
    binary_global = blurred_image < global_thresh
    ax3.imshow(binary_global,cmap='gray')
    ax3.set_title("Leads 13")
    ax3.axis('off')
    fig3.savefig('Preprossed_Leads_13_figure.png')

    my_expander2 = st.expander(label='PREPROCESSED LEAD')
    with my_expander2:
      st.image('Preprossed_Leads_1-12_figure.png')
      st.image('Preprossed_Leads_13_figure.png')
    
    """#### **EXTRACTING SIGNALS(1-13)**"""
    fig4 , ax4 = plt.subplots(4,3)
    fig4.set_size_inches(10, 10)
    x_counter=0
    y_counter=0
    for x,y in enumerate(Leads[:len(Leads)-1]):
      #converting to gray scale
      grayscale = color.rgb2gray(y)
      #smoothing image
      blurred_image = gaussian(grayscale, sigma=0.9)
      #thresholding to distinguish foreground and background
      #using otsu thresholding for getting threshold value
      global_thresh = threshold_otsu(blurred_image)

      #creating binary image based on threshold
      binary_global = blurred_image < global_thresh
      #resize image
      binary_global = resize(binary_global, (300, 450))
      #finding contours
      contours = measure.find_contours(binary_global,0.8)
      contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
      for contour in contours:
        if contour.shape in contours_shape:
          test = resize(contour, (255, 2))
      if (x+1)%3==0:
        ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
        ax4[x_counter][y_counter].axis('image')
        ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
        x_counter+=1
        y_counter=0
      else:
        ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
        ax4[x_counter][y_counter].axis('image')
        ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
        y_counter+=1
    
      #scaling the data and testing
      lead_no=x
      scaler = MinMaxScaler()
      fit_transform_data = scaler.fit_transform(test)
      Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
      Normalized_Scaled=Normalized_Scaled.T
      #scaled_data to CSV
      if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
        Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
      else:
        Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
    
    fig4.savefig('Contour_Leads_1-12_figure.png')
    my_expander3 = st.expander(label='CONOTUR LEADS')
    with my_expander3:
      st.image('Contour_Leads_1-12_figure.png')

    """#### **CONVERTING TO 1D SIGNAL**"""    
    #lets try combining all 12 leads
    test_final=pd.read_csv('/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/Scaled_1DLead_1.csv')
#     location= '/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/'

#     allowed_files = [
#     f"Scaled_1DLead_{i}.csv" for i in range(1, 13)
#     ]
#     st.write("Shape BEFORE PCA:", test_final.shape)
#     #test_final = pd.DataFrame()
#     # ✅ 2. Match Kaggle logic: skip Lead 13 and specific unwanted files
#     for file in natsorted(os.listdir(location)):
#       if file in allowed_files:
#           df = pd.read_csv(os.path.join(location, file))

#           # Drop 'Unnamed: 0' if exists
#           if 'Unnamed: 0' in df.columns:
#               df.drop(columns=['Unnamed: 0'], inplace=True)

#           # Merge
#           test_final = pd.concat([test_final, df], axis=1, ignore_index=True)

#           # Drop last column after concat
#           #test_final.drop(columns=test_final.columns[-1], inplace=True)

#     if 255 in test_final.columns:
#       test_final.drop(columns=[255], inplace=True)
#     st.write("test_final shape BEFORE PCA:", test_final.shape)

#     """#### **PASS TO ML MODEL FOR PREDICTION**"""
#         # ✅ Load trained PCA
#     pca = joblib.load("/Users/sanglapacharyya/Downloads/PCA_ECG.pkl")
#     expected_features = pca.n_features_in_

# # ✅ Transform test data to expected size (400 features)
#     test_final = test_final.iloc[:, :expected_features]
#     test_final_reduced = pca.transform(test_final)

    # ✅ Define correct folder
    location = '/Users/sanglapacharyya/joy/Cardiovascular-Detection-by-ECG-images/'

    # ✅ Load all leads 1–12 into a SINGLE row
    test_final = pd.DataFrame()
    for i in range(1, 12):
        file_path = os.path.join(location, f'Scaled_1DLead_{i}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Drop auto-indexed columns if needed
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # ✅ Ensure shape (1, N) before concatenating
            df = df.reset_index(drop=True)

            # ✅ Concatenate horizontally
            test_final = pd.concat([test_final, df], axis=1)

    st.write("✅ Combined shape before PCA:", test_final.shape)

    # ✅ Now verify expected dims BEFORE slicing
    pca = joblib.load("/Users/sanglapacharyya/Downloads/PCA_ECG.pkl")
    expected_features = pca.n_features_in_

    if test_final.shape[1] != expected_features:
        st.write(f"⚠ Feature mismatch! Got {test_final.shape[1]}, expected {expected_features}")

    # ✅ Trim excess if too many columns
    # If there's exactly ONE extra column, drop it
    if test_final.shape[1] > expected_features:
        test_final = test_final.iloc[:, :expected_features]
    elif test_final.shape[1] < expected_features:
        st.error(f"Not enough features: got {test_final.shape[1]}, expected {expected_features}")
        st.stop()

    test_final_reduced = pca.transform(test_final)

  



    from joblib import load

    model = joblib.load("/Users/sanglapacharyya/Downloads/Heart_Disease_Prediction_using_ECG_new.pkl", mmap_mode='r')

  

# ✅ Pass reduced features to model
    result = model.predict(test_final_reduced)

    if result[0] == 0:
      st.write("You ECG corresponds to Myocardial Infarction")
    
    if result[0] == 1:
      st.write("You ECG corresponds to Abnormal Heartbeat")
    
    if result[0] == 2:
      st.write("Your ECG is Normal")
    
    if result[0] == 3:
      st.write("You ECG corresponds to History of Myocardial Infarction")
    
  else:
    st.write("Sorry Our App won't be able to parse this image format right now!!!. Pls check the image input sample section for supported images")
  
  
  st.write("Model classes:", model.classes_)
