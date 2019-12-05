import argparse
import os
import time

import utils
import utilsT
import models


def post_train(model=None,
               chosen_diseases=None,
               run_name=None,
               base_dir=".",
               test_max_images=None,
               batch_size=5,
               experiment_mode="debug",
              ):
    device = utilsT.get_torch_device()
    print("Using device: ", device)

    dataset_dir = os.path.join(base_dir, "dataset")

    
    if model is None or chosen_diseases is None:
        if run_name is None:
            print("Can't run post: run_name not provided")
            return

        print("Loading model...")
        model, model_name, optimizer, opt_name, chosen_diseases = models.load_model(base_dir,
                                                                                    run_name,
                                                                                    experiment_mode=experiment_mode,
                                                                                    device=device,
                                                                                   )
        model.train(False)
    
    print("Loading test dataset...")
    test_dataset, test_dataloader = utilsT.prepare_data(dataset_dir,
                                                        "test",
                                                        chosen_diseases,
                                                        batch_size,
                                                        max_images=test_max_images,
                                                       )
    
    
    
    ## Save CM with names
    n_test_images = test_dataset.size()[0]
    n_diseases = len(chosen_diseases)
    
    print("Predicting all...")
    test_predictions, test_gts, test_image_names = utils.predict_all(model,
                                                                     test_dataloader,
                                                                     device,
                                                                     n_test_images,
                                                                     n_diseases)

    print("Calculating CM...")
    test_cms_names = utils.calculate_all_cms_names(test_predictions, test_gts, test_image_names, chosen_diseases)

    fname = os.path.join(utils.CMS_DIR,
                         experiment_mode,
                         run_name + "_test_names",
                        )
    utils.save_cms_names(test_cms_names, fname, chosen_diseases)



def parse_args():
    parser = argparse.ArgumentParser(description="Run after train")
    
    parser.add_argument("run_name", type=str, help="Run name to load the model")
    parser.add_argument("--base-dir", type=str, default="/mnt/data/chest-x-ray-8", help="Base folder")
    parser.add_argument("--test-max-images", type=int, default=None, help="Max amount of test images")
    parser.add_argument("--non-debug", action="store_true", help="If present set non-debugging mode")
    
    
    args = parser.parse_args()
    
    if not args.non_debug:
        args.experiment_mode = "debug"
    else:
        args.experiment_mode = ""
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    start = time.time()
    post_train(run_name=args.run_name,
               base_dir=args.base_dir,
               test_max_images=args.test_max_images,
               experiment_mode=args.experiment_mode,
              )
    end = time.time()
    print("Total time: ", utils.duration_to_str(end - start))
    print("="*50)