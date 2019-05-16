

def print_params_CNN(  ):
    
    print( 'Param          :', paramName[paramNum] )
    print( 'file           :', model_file )
    print( 'RandomSeed     :', RandomSeed )
    print( 'trainSize      :', trainSize )
    print( 'LHS            :', LHS )
    if( LHS ):
        print( 'Nbins_LHS      :', Nbins_LHS )
    print( 'epochs         :', epochs )
    print( 'batch_size     :', batch_size )
    print( 'DATABASE       :', DATABASE )
    print( 'validation     :', validation )
    print( 'all4           :', all4           )
    print( 'reduce_LC      :', reduce_LC      )
    print( 'use_dropout    :', use_dropout    )
    print( 'substract_mean :', substract_mean )
    print( 'apply_gauss    :', apply_gauss    )
    print( 'CNN loss       :', loss )
    print( 'CNN optimizer  :', optimizer )
    print( 'LR factor      :', factor)
    print( 'LR patience    :', patience)

    ### Variables not define in all parameter file!!
    try:
        print( 'LeackyRelu     :',LeackyRelu_alpha )
    except:
        LeackyRelu_alpha = 0
        print( 'LeackyRelu     :',LeackyRelu_alpha )

    try:
        print( 'batchNorm      :',batchNorm )
    except:
        batchNorm = False
        print( 'batchNorm      :',batchNorm )