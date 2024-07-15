import streamlit as st
from PIL import Image
import io
from azure.storage.blob import BlobServiceClient
import requests
import json
import pandas as pd
import re
import uuid
# import cv2

def main():
    st.title("Document LLM OCR PoC")
    st.subheader("Passport and Bank Statement")
    
    st.header("Upload Image")
    container_name = "phi3"
    storage_account_name = "jhdemo1storageaccount"
    storage_account_key = st.secrets["storage_account_key"]
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    

    # st.header("Or Take a Photo")
    # run_camera = st.button("Open Camera")
    
    # if run_camera:
    #     run_webcam()
    
    st.header("Result")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize and compress the image
        if (len(uploaded_file.getvalue()) / 1024 > 80):
            image = resize_and_compress_image(image)
        else:
            image = uploaded_file.getvalue()

        blob_url = upload_to_azure_blob(uploaded_file.name, image, container_name, storage_account_name, storage_account_key)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if blob_url:
            # st.success(f"File uploaded successfully. URL: {blob_url}")
            result = phi3_ocr_api(blob_url)
            # st.success(f"Result:")
            # st.success(result)
            result = clean_json_string(result)
            # st.success(result)
            df = pd.json_normalize(json.loads(result))
            st.table(df)
        else:
            st.error("Failed to upload file.")

        # process_image(image)
    
# def run_webcam():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Could not open camera.")
#         return
    
#     st.warning("Press 's' to take a screenshot")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Error: Failed to grab frame.")
#             break
        
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.flip(frame, 1)
        
#         frame = cv2.resize(frame, (640, 480))
        
#         cv2.imshow('Camera', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             cv2.imwrite('screenshot.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
    
#     st.success("Photo taken!")

def resize_and_compress_image(image, max_size=(640, 480), quality=90):
    image.thumbnail(max_size)
    
    # Create a buffer to hold the image data
    img_byte_array = io.BytesIO()
    
    # Save the image to the buffer
    image.save(img_byte_array, format='JPEG', quality=quality, optimize=True)
    
    return img_byte_array.getvalue()

# Function to upload file to Azure Blob Storage
def upload_to_azure_blob(filename, image_data, container_name, storage_account_name, storage_account_key):
    try:
        # Initialize the BlobServiceClient using account key
        connect_str = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # Get a client to interact with a specific container
        container_client = blob_service_client.get_container_client(container_name)
        
        # Upload the file to blob storage
        # Generate a random UUID
        uuid_value = uuid.uuid4()
        uuid_string = str(uuid_value)
        blob_client = container_client.upload_blob(name=filename + uuid_string, data=image_data)

        # Get the URL of the uploaded blob
        blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{filename + uuid_string}"
        
        return blob_url
    
    except Exception as e:
        st.error(f"Error uploading file to Azure Blob Storage: {e}")
        return None

def phi3_ocr_api(blob_url):

    url = st.secrets["phi3_endpoint"]

    payload = json.dumps({
        "model": "microsoft/Phi-3-vision-128k-instruct",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "You are a passport subject expert. Please read the passport and output the information on the passport in json format with the following fields: Passport type, Passport number, Country, Name, Given name, Gender, Place of Birth, Date of Birth, Date of issue, date of expiry, Place of issue. If the corresponding information is not available, output N/A instead. You must only output the json data and no any other data!"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": blob_url,
                    "detail": "high"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    })

    headers = {
        'Authorization': 'Bearer ' + st.secrets["phi3_api_key"],
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json().get("choices")[0].get("message").get("content")
    else:
        return None

def clean_json_string(text):
    matches = text.split('}')
    # st.success(matches)
    
    return matches[0] + '}' 

def process_image(image):
    st.subheader("Image Processing Result")
    
    # Perform some basic image processing here if needed
    # For example, you can apply filters, resize, etc.
    processed_image = image  # Placeholder, replace with actual processing
    
    st.image(processed_image, caption="Processed Image", use_column_width=True)

    
if __name__ == '__main__':
    main()
