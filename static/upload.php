<?php
$targetDir = "uploads/";
$fileName = basename($_FILES["video"]["name"]);
$targetFilePath = $targetDir . $fileName;
$fileType = pathinfo($targetFilePath, PATHINFO_EXTENSION);

if(isset($_POST["submit"]) && !empty($_FILES["video"]["name"])){
    // 허용된 파일 포맷들
    $allowTypes = array('mp4','avi','mov');
    if(in_array($fileType, $allowTypes)){
        // 파일 업로드 시도
        if(move_uploaded_file($_FILES["video"]["tmp_name"], $targetFilePath)){
            echo "The file ".$fileName. " has been uploaded.";
        } else{
            echo "Sorry, there was an error uploading your file.";
        }
    } else{
        echo "Sorry, only MP4, AVI, MOV files are allowed to upload.";
    }
} else{
    echo "Please select a file to upload.";
}
?>
