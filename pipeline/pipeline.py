import kfp
from kfp import dsl

@dsl.pipeline(
    name='Vietnamese Flowers Classification',
    description='Pipeline for 5 flower types using SVM/Random Forest'
)
def flower_pipeline(
    raw_data_dir: str = '/app/data',
    augment_num_rotations: int = 3,
    model_type: str = 'svm',
    train_epochs: int = 1  # không dùng nhưng giữ để tương thích
):
    # Bước 1: Augmentation (xoay ảnh) trên tập train
    augment = dsl.ContainerOp(
        name='augment',
        image='<ACR_NAME>.azurecr.io/flower-augment:v1',
        command=['python', 'augment.py'],
        arguments=[
            '--input_dir', raw_data_dir + '/train',
            '--output_dir', '/app/augmented_train',
            '--num_rotations', str(augment_num_rotations)
        ]
    )
    
    # Bước 2: Huấn luyện trên dữ liệu đã augment
    train = dsl.ContainerOp(
        name='train',
        image='<ACR_NAME>.azurecr.io/flower-train:v1',
        command=['python', 'train.py'],
        arguments=[
            '--data_dir', '/app/augmented_train',
            '--model_type', model_type,
            '--output_dir', '/model'
        ],
        file_outputs={'model_dir': '/model'}
    ).after(augment)
    
    # Bước 3: Đánh giá trên tập test gốc
    evaluate = dsl.ContainerOp(
        name='evaluate',
        image='<ACR_NAME>.azurecr.io/flower-evaluate:v1',
        command=['python', 'evaluate.py'],
        arguments=[
            '--data_dir', raw_data_dir + '/test',
            '--model_dir', train.outputs['model_dir']
        ]
    ).after(train)
    
    # Bước 4: Serve (tuỳ chọn, có thể bỏ nếu không cần)
    serve = dsl.ContainerOp(
        name='serve',
        image='<ACR_NAME>.azurecr.io/flower-serve:v1',
        arguments=[
            '--model_dir', train.outputs['model_dir'],
            '--port', '5000'
        ]
    ).after(evaluate)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(flower_pipeline, 'flower_pipeline.yaml')