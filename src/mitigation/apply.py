""" def apply_mitigations_preprocessing(dataset, config):

    mitigation = config.get("mitigation", {})

    if mitigation.get("balancing", {}).get("enabled", False):
        dataset = balance_dataset(
            dataset,
            strategy=mitigation["balancing"]["strategy"],
        )

    if mitigation.get("gender_augmentation", {}).get("enabled", False):
        dataset = augment_with_gender_swap(dataset)

    return dataset """


##def apply_inprocessing