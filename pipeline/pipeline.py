import kfp
from kfp import dsl

@dsl.pipeline(
    name='Orchid Classification Pipeline',
    description='Train SVM/Random Forest using handcrafted features'
)
def orchid_pipeline(
    train_data_dir: str = '/app/data/train',
    valid_data_dir: str = '/app/data/valid',
    model_type: str = 'svm',
    model_output_dir: str = '/model'
):
    train = dsl.ContainerOp(
        name='train',
        image='<ACR_NAME>.azurecr.io/orchid-train:v1',
        arguments=[
            '--data_dir', train_data_dir,
            '--model_type', model_type,
            '--output_dir', model_output_dir
        ],
        file_outputs={'model_dir': model_output_dir}
    )
    
    evaluate = dsl.ContainerOp(
        name='evaluate',
        image='<ACR_NAME>.azurecr.io/orchid-evaluate:v1',
        arguments=[
            '--data_dir', valid_data_dir,
            '--model_dir', train.outputs['model_dir']
        ]
    ).after(train)
    
    # (Tuỳ chọn) serve - có thể chạy sau hoặc tách riêng
    serve = dsl.ContainerOp(
        name='serve',
        image='<ACR_NAME>.azurecr.io/orchid-serve:v1',
        arguments=[
            '--model_dir', train.outputs['model_dir'],
            '--port', '5000'
        ]
    ).after(evaluate)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(orchid_pipeline, 'orchid_pipeline.yaml')