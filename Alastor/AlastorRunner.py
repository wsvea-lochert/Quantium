from AlastorTrainer import AlastorTrainer

trainer = AlastorTrainer(json_folder='dataset2-flip/json/', image_folder='dataset2-flip/train/', kp_def='dataset2-flip/dataset_head.csv',
                         checkpoint_dir='checkpoints/', save_dir='models/', log_dir='logs/')
