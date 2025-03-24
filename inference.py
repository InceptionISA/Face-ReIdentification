from src.face_recognition_individual import FaceRecognitionIndividual

# Initialize face recognition pipeline
voting_sys = FaceRecognitionIndividual()


# Test the pipeline on special images
# results = face_recognition.test_on_images()
# print(results)


# Generate submissions using all models
all_submissions = voting_sys.run_pipeline()
print("All submissions generated",'\n' ,all_submissions)