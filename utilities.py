import matplotlib.pyplot as plt
def get_product_features(dataset):
    product_features = dataset.map(lambda x:{
        "itemid": x["itemid"],
        "timestamp": x["timestamp"],
        "property_1": x["property_1"],
        "property_2": x["property_2"],
        "property_3": x["property_3"],
        "property_4": x["property_4"],
        "property_5": x["property_5"],
        "property_6": x["property_6"],
        "property_7": x["property_7"],
        "property_8": x["property_8"],
        "property_9": x["property_9"],
        "property_10": x["property_10"],
        "property_11": x["property_11"],
        "available": x["available"],
        "categoryid": x["categoryid"],
        "parent_level_1": x["parent_level_1"],
        "parent_level_2": x["parent_level_2"],
        "parent_level_3": x["parent_level_3"],
        "parent_level_4": x["parent_level_4"],
        "parent_level_5": x["parent_level_5"]
        })
    return product_features


def visualisation(model):
    
     # Plot Accuracy and Loss (Traning and Validation)
    # Accuracy
    plt.plot(model.history.history["factorized_top_k/top_100_categorical_accuracy"], label="train_accuracy")
    plt.plot(model.history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="val_accuracy")
    plt.legend()
    plt.title("Train and Validation Accuracy")
    plt.show()
    plt.close()

    # Loss
    plt.plot(model.history.history["total_loss"], label="train_loss")
    plt.plot(model.history.history["val_total_loss"], label="val_loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.title("Train and Validation Loss")
    plt.show()
    plt.close()
