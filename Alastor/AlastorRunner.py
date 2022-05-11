from AlastorTrainer import AlastorTrainer

trainer = AlastorTrainer(json_folder='C:/Users/William/Documents/C3P0 datasets/dataset2/advanced/split/',
                         image_folder='C:/Users/William/Documents/C3P0 datasets/dataset2/advanced/img/',
                         kp_def='C:/Users/William/Documents/C3P0 datasets/dataset2/dataset_head.csv',
                         checkpoint_dir='C:/Users/William/Documents/git/Quantium/Alastor/checkpoints/',
                         save_dir='C:/Users/William/Documents/git/Quantium/Alastor/models/',
                         log_dir='C:/Users/William/Documents/git/Quantium/Alastor/logs/')
trainer.run()
