from face_reidentification import FaceRecognition

# Initialize face recognition pipeline
face_recognition = FaceRecognition()


# Test the pipeline on special images
# results = face_recognition.test_on_images()
# print(results)


# Generate submissions using all models
all_submissions = face_recognition.run_pipeline()
