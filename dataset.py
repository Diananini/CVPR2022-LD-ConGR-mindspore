from datasets.hcigesture import HCIGesture

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['hcigesture']

    if opt.train_validate:
        subset = ['training', 'validation']
    else:
        subset = 'training'

    if opt.dataset == 'hcigesture':
        training_data = HCIGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            iscrop=opt.iscrop,
            isKeyframes=opt.isKeyframes)
        
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['hcigesture']

    if opt.dataset == 'hcigesture':
        validation_data = HCIGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            iscrop=opt.iscrop,
            isKeyframes=opt.isKeyframes,
            n_samples_for_each_video=opt.n_val_samples)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['hcigesture']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'hcigesture':
        test_data = HCIGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            iscrop=opt.iscrop,
            isKeyframes=opt.isKeyframes,
            n_samples_for_each_video=opt.n_val_samples)
    return test_data
