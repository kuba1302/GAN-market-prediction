import pickle
import dill
from models.gan.gan import generator, discriminator, StockTimeGan
from pathlib import Path
import os
import numpy as np

if __name__ == "__main__":
    MODEL_VERSION = "0.6"
    load_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
    save_path = Path(os.path.abspath("")).parents[0] / "models" / "gan" / "versions"

    with open(load_path / "data.pickle", "rb") as test:
        data = pickle.load(test)
    print(
        '---------------------------- '
        f'TRAIN DATA SHAPE: {data["X_list_train"].shape}'
        ' ----------------------------'
    )
    generator = generator(data["X_list_train"].shape[1], data["X_list_train"].shape[2])
    discriminator = discriminator((31, data["Y_preds_real_list_train"].shape[1]))
    gan = StockTimeGan(
        generator=generator, discriminator=discriminator, checkpoint_directory=save_path
    )
    train_history = gan.train(
        data["X_list_train"],
        data["Y_preds_real_list_train"],
        data["Y_whole_real_list_train"],
        epochs=500,
    )
    test_preds = gan.predict(data["X_list_test"])
    model_data = {
        "train_history": train_history,
        "test_preds": np.array(test_preds),
        "actual_values": np.array(data["Y_preds_real_list_test"]),
    }
    gan.save_generator(save_path / f"model_{MODEL_VERSION}_class")
    with open(save_path / f"model_{MODEL_VERSION}.pickle", "wb") as handle:
        pickle.dump(model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
