The dataset was created and reviewed using a combination of publicly available datasets and real-world recordings, covering a wide range of vehicle faults, crashes, emergency sirens (police, ambulance, fire truck),

wild animal sounds, car and truck horns, and other environmental road sounds. This approach ensures a diverse and realistic dataset that enhances model performance in detecting road-related events.
________________________________________
1. Data Collection and Annotation
   
•	We carefully selected publicly available datasets that include real traffic scenarios and vehicle fault cases.

•	Additionally, we extracted relevant frames and sequences from YouTube videos, ensuring a diverse representation of traffic conditions and vehicle behaviors.

•	Each data sample was manually labeled based on predefined criteria, focusing on vehicle states, traffic interactions, and specific fault conditions.
________________________________________
2. Expert Review and Validation
   
•	To enhance reliability, domain experts with extensive experience in automotive engineering and machine learning reviewed the dataset.

•	The experts cross-checked and validated the labels to ensure accuracy and consistency with real-world vehicle behaviors.
________________________________________
3. Publicly Available Datasets Referenced
   
We utilized multiple datasets containing wild animal sounds, vehicle faults, and environmental noises to build a comprehensive dataset. The key datasets referenced include:

•	[FSC22 Dataset]: A collection focused on various sound categories, including vehicle sounds and environmental noises, useful for sound classification models [1].

•	[Google AudioSet]: A large-scale collection of audio data across thousands of categories, aimed at improving sound classification models [2].

•	[Audio Data]: A dataset containing diverse audio clips across various categories, useful for developing classification models [3].

•	[Sound Classification of Animal Voice]: A dataset containing sounds from different animals, useful for animal sound classification tasks [4].

•	[DCASE 2024 Challenge]: A dataset designed for the DCASE 2024 challenge, covering environmental sound classification tasks [5].

•	[UrbanSound8K Dataset]: Contains 8,732 labeled sound excerpts from urban environments, categorized into 10 classes such as car horns and sirens [6].

•	[AudioSet by Google Research]: A vast dataset with over 2 million human-labeled 10-second sound clips spanning thousands of audio categories [7].

•	[Vehicle Sounds Dataset]: Contains various vehicle sounds useful for training models focused on transportation-related sound classification [8].
________________________________________
4. Vehicle Faults Categorized in the Dataset
   
The dataset includes multiple vehicle faults, ensuring coverage of mechanical, transmission, braking, and steering issues. The specific fault categories are:

✅ Transmission and Drivetrain Faults

•	Bad CV Joint
•	Bad Transmission
•	Bad Wheel Bearing
•	Universal Joint Failure / Steering Rack Failure
•	Turning Front-End Clicking (Bad CV Axle)

✅ Engine and Exhaust Faults

•	Engine Chirping / Squealing Belt
•	Engine Misfire
•	Engine Rattle Noise
•	Flooded Engine
•	Fuel Pump Cartridge Fault
•	Knocking
•	Lifter Ticking
•	Loose Exhaust Shield
•	Muffler Running Loud (Exhaust Leak)
•	Pre-Ignition
•	Seized Engine
•	Thrown Rod
•	Vacuum Leak

✅ Suspension and Steering Faults

•	Clunking Over Bumps (Bad Stabilizer Link Noise)
•	Steering Groaning / Whining (Low Power Steering Fluid)
•	Steering Noise
•	Strut Mount Failure
•	Suspension Arm Fault

✅ Cooling System and Fan Faults

•	Radiator Fan Failure

✅ Braking System Faults

•	Squeaky Belt
•	Squeaky Brake / Grinding Brake
________________________________________
5. Car Crash References
   
To include realistic car crash scenarios, we incorporated real-world accident recordings from YouTube [9]–[37].
________________________________________
6. Animal Sound References
   
To accurately reflect real-world environmental conditions, we included animal sounds from various real-world recordings available on YouTube [38]–[50].
________________________________________
7. Additional Car Fault References
   
We supplemented our dataset with real-world recordings of car faults obtained from Instagram and YouTube [51]–[72].

________________________________________
References 

[1] Kaggle, "FSC22 Dataset," Available: https://www.kaggle.com/datasets/irmiot22/fsc22-dataset.

[2] Kaggle, "Google Audioset," Available: https://www.kaggle.com/datasets/akela91/google-audioset.

[3] Kaggle, "Audio Data," Available: https://www.kaggle.com/datasets/ivanj0/audiodata.

[4] Kaggle, "Sound Classification of Animal Voice," Available: https://www.kaggle.com/datasets/rushibalajiputthewad/sound-classification-of-animal-voice.

[5] DCASE Community, "DCASE 2024 Challenge," Available: https://dcase.community/challenge2024/index#task1.

[6] UrbanSound, "UrbanSound8K Dataset," Available: https://urbansounddataset.weebly.com/urbansound8k.html.

[7] Google Research, "AudioSet," Available: https://research.google.com/audioset/.

[8] Kaggle, "Vehicle Sounds Dataset," Available: https://www.kaggle.com/datasets/janboubiabderrahim/vehicle-sounds-dataset.

[9] YouTube, "Car Crash," Available: https://youtu.be/qCzYyjbfmZ4.

[10] YouTube, " Car Crash," Available: https://youtu.be/iy8TyBFM17k.

[11] YouTube, " Car Crash," Available: https://youtube.com/shorts/TQMX_nIRoBY.

[12] YouTube, " Car Crash," Available: https://youtu.be/WoIbrsF8w_0.

[13] YouTube, " Car Crash," Available: https://www.youtube.com/shorts/LMN7zOgx6-I.

[14] YouTube, " Car Crash," Available: https://youtu.be/sRJ-Ly8gywk.

[15] YouTube, " Car Crash," Available: https://youtu.be/ewayVSFOtsI.

[16] YouTube, " Car Crash," Available: https://youtube.com/shorts/0qgKpNnhoQ0.

[17] YouTube, " Car Crash," Available: https://youtube.com/shorts/4xX6wQv3jaY.

[18] YouTube, " Car Crash," Available: https://youtube.com/shorts/Pl0sKYYE9bk.

[19] YouTube, " Car Crash," Available: https://youtu.be/KWM8bIpVq5o.

[20] YouTube, " Car Crash,"Available: https://youtube.com/shorts/hlisAvTFBtU.

[21] YouTube, " Car Crash," Available: https://youtu.be/bE2F5BIQEjQ.

[22] YouTube, " Car Crash," Available: https://youtu.be/Wfx1pH5MD5g.

[23] YouTube, " Car Crash," Available: https://youtu.be/DbXtQBgunSI.

[24] YouTube, " Car Crash," Available: https://youtu.be/AFWWqUOShj8.

[25] YouTube, " Car Crash," Available: https://youtu.be/Gp3vBR4fKao.

[26] YouTube, " Car Crash," Available: https://youtube.com/shorts/_GeKYYCMcWk.

[27] YouTube, " Car Crash," Available: https://youtube.com/shorts/GEl2SB6tjyM.

[28] YouTube, " Car Crash," Available: https://youtu.be/mrpg_n1iXNU.

[29] YouTube, " Car Crash," Available: https://youtu.be/V94DWRKssZQ.

[30] YouTube, " Car Crash," Available: https://youtu.be/zn66td1ZPuc.

[31] YouTube, " Car Crash," Available: https://youtu.be/NoSTTF2fB98.

[32] YouTube," Car Crash," Available: https://youtu.be/FlxHZS5cMKI.

[33] YouTube, " Car Crash," Available: https://youtu.be/7ThTci70350.

[34] YouTube, " Car Crash," Available: https://youtu.be/3LL82WkOujM.

[35] YouTube, " Car Crash," Available: https://youtube.com/shorts/UAi46eie2j0.

[36] YouTube, " Car Crash," Available: https://youtu.be/i7ss31_aado.

[37] YouTube, " Car Crash," Available: https://youtu.be/FL9AX9WTHnY.

[38] YouTube, "Animal Sounds ," Available: https://www.youtube.com/watch?v=BV-AeTn0c6c.

[39] YouTube, "Animal Sounds," Available: https://www.youtube.com/watch?v=mX8LegY5sEc.

[40] YouTube, "Animal Sounds," Available: https://www.youtube.com/shorts/A0vje1Wg2-4.

[41] YouTube, "Animal Sounds," Available: https://www.youtube.com/shorts/x-1QR33so_g.

[42] YouTube, "Animal Sounds," Available: https://youtube.com/shorts/NRzy0LuUmRQ.

[43] YouTube, "Animal Sounds," Available: https://youtube.com/shorts/qMHiNvaMlDE.

[44] YouTube, "Animal Sounds," Available: https://youtu.be/uvVvlNT8Y8I.

[45] YouTube, "Animal Sounds," Available: https://youtube.com/shorts/EE8LUWmi9E4.

[46] YouTube, "Animal Sounds,"  Available: https://youtube.com/shorts/nuC0Jd_fiuk.

[47] YouTube, "Animal Sounds ," Available: https://youtu.be/iST-Brwk0LQ.

[48] YouTube, "Animal Sounds ," Available: https://youtube.com/shorts/jYEWi88Ij1g.

[49] YouTube, "Animal Sounds ,"Available: https://youtu.be/eFoBwHU2Fh0.

[50] YouTube, "Animal Sounds ," Available: https://youtube.com/shorts/qm-kMiTw-mE.

[51] Instagram, "Reel Video: Car Faults," Available: https://www.instagram.com/reel/DAtBzrXNjhf/.

[52] YouTube, "Short Video: Car Faults," Available: https://youtube.com/shorts/XW8qx0WPYWA.

[53] YouTube, "Short Video: Car Faults," Available: https://youtube.com/shorts/ETYaJUNdbqM.

[54] YouTube, "Video: Car Faults," Available: https://youtu.be/ChhHs0ZBfWU.

[55] YouTube, "Video: Car Faults," Available: https://youtu.be/_SnOrnmFpow.

[56] YouTube Video: Car Faults," Available: https://www.youtube.com/watch?v=Bad9JIBFoOI.

[57] YouTube, "Short Video: Car Faults," Available: https://www.youtube.com/shorts/dXDcVqmQa_I.

[58] "YouTube Video: Car Faults," Available: https://www.youtube.com/watch?v=rZGpjvjaoJA.

[59] YouTube, "Short Video: Car Faults," Available: https://youtube.com/shorts/qWsZnBamXa8.

[60] YouTube, "Video: Car Faults," Available: https://youtu.be/vsSuoCbnKRo.

[61] YouTube, "Video: Car Faults," Available: https://youtu.be/iE0DFNX1h2Q.

[62] YouTube, "Video: Car Faults," Available: https://youtu.be/4X3RW7u14Rc.

[63] YouTube, "Short Video: Car Faults," Available: https://youtube.com/shorts/-p-9FaXh2IE.

[64] YouTube, "Video: Car Faults," Available: https://youtu.be/GspWr69V96Y.

[65] YouTube, "Short Video: Car Faults," Available: https://youtube.com/shorts/N3259hdbhUc.

[66] YouTube, "Short Video: Car Faults," Available: https://youtube.com/shorts/Yy3l0pToRzc.

[67] YouTube, "Short Video: Car Faults," Available: https://youtube.com/shorts/yowRYFa2gjw.

[68] YouTube, "Video: Car Faults," Available: https://youtu.be/bS-gx5IY5Sc.

[69] YouTube, "Video: Car Faults," Available: https://youtu.be/0q9rV0UvZUA.

[70] YouTube, "Video: Car Faults," Available: https://youtu.be/ss1ge3VA_a8.

[71] YouTube, "Video: Car Faults," Available: https://youtu.be/rx7T31LleQw.

[72] YouTube, "Video: Car Faults," Available: https://youtu.be/_NAlRq6HaOA.
