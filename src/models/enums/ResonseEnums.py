from enum import Enum

class ResponseMessage(Enum):
    FILETYPENOTSUPPORTED = "File type not supported"
    FILEUPLOADSUCCESS = "File uploaded successfully"
    FILEUPLOADFAILED = "File upload failed"
    FILESIZEEXCEEDED = "File size exceeded"
    FileNotFoundError = "File not found"
    FILEPROCESSINGFAILED = "File processing failed"
    FILEPROCESSINGSUCCESS = "File processing success"
    PROJECTCREATED = "Project created successfully"
    PROJECTNOTFOUND = "Project not found"
    PROJECTDELETED = "Project deleted successfully"
    PROJECTUPDATED = "Project updated successfully"
    NOFILEERROR = "No file uploaded"
    FILENOTEXIST = "File does not exist"
    PERSONNOTEXIST = "Person does not exist"
    PERSONCREATED = "Person created successfully"
    PERSONDELETED = "Person deleted successfully"
    PERSONUPDATED = "Person updated successfully"
    FACEPROCESSINGFAILED = "Face processing failed"
    INTERNALSERVERERROR = "Internal server error"

    VECTORDB_COLLECTION_RETRIEVED = "Vector DB collection retrieved successfully"
  