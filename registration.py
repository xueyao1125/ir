import itk

fixedImageFile = 'D:/peizhunDATA/fixed.mhd'
movingImageFile = 'D:/peizhunDATA/moving.mhd'
outputImageFile = 'D:/peizhunDATA/output.png'
differenceImageAfterFile = 'D:/peizhunDATA/differenceImageAfter.png'
differenceImageBeforeFile = 'D:/peizhunDATA/differenceImageBefore.png'


# read image, define image dimension and type
PixelType = itk.ctype('float')
fixedImage = itk.imread(fixedImageFile, PixelType)
movingImage = itk.imread(movingImageFile, PixelType)
Dimension = fixedImage.GetImageDimension()
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]


TransformType = itk.TranslationTransform[itk.D, Dimension]
initialTransform = TransformType.New()


# using step gradiant to construct optimizer
optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
    LearningRate=4,
    MinimumStepLength=0.001,
    RelaxationFactor=0.5,
    NumberOfIterations=200)


# similarity measure
metric = itk.MeanSquaresImageToImageMetricv4[
    FixedImageType,
    MovingImageType].New()



registration = itk.ImageRegistrationMethodv4.New(FixedImage=fixedImage,
                                                 MovingImage=movingImage,
                                                 Metric=metric,
                                                 Optimizer=optimizer,
                                                 InitialTransform=initialTransform)



# initialize parameters
movingInitialTransform = TransformType.New()
initialParameters = movingInitialTransform.GetParameters()
initialParameters[0] = 0
initialParameters[1] = 0
movingInitialTransform.SetParameters(initialParameters)
registration.SetMovingInitialTransform(movingInitialTransform)
identityTransform = TransformType.New()
identityTransform.SetIdentity()
registration.SetFixedInitialTransform(identityTransform)
registration.SetNumberOfLevels(1)
registration.SetSmoothingSigmasPerLevel([0])
registration.SetShrinkFactorsPerLevel([1])


registration.Update()


# get parameters from direction x and y
transform = registration.GetTransform()
finalParameters = transform.GetParameters()
translationAlongX = finalParameters.GetElement(0)
translationAlongY = finalParameters.GetElement(1)


numberOfIterations = optimizer.GetCurrentIteration()


bestValue = optimizer.GetValue()


# output
print("Result = ")
print(" Translation X = " + str(translationAlongX))
print(" Translation Y = " + str(translationAlongY))
print(" Iterations    = " + str(numberOfIterations))
print(" Metric value  = " + str(bestValue))


CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
outputCompositeTransform = CompositeTransformType.New()
outputCompositeTransform.AddTransform(movingInitialTransform)
outputCompositeTransform.AddTransform(registration.GetModifiableTransform())


resampler = itk.ResampleImageFilter.New(Input=movingImage,
                                        Transform=outputCompositeTransform,
                                        UseReferenceImage=True,
                                        ReferenceImage=fixedImage)
resampler.SetDefaultPixelValue(100)   # 设置默认灰度值,用来 "突出" 显示映射后在浮动图像之外的区域



OutputPixelType = itk.ctype('unsigned char')
OutputImageType = itk.Image[OutputPixelType, Dimension]



caster = itk.CastImageFilter[FixedImageType,
                             OutputImageType].New(Input=resampler)



writer = itk.ImageFileWriter.New(Input=caster, FileName=outputImageFile)
writer.SetFileName(outputImageFile)
writer.Update()


# comparison between fixed and moving image
difference = itk.SubtractImageFilter.New(Input1=fixedImage,
                                         Input2=resampler)


# itk.RescaleIntensityImageFilter to adjust intensity
intensityRescaler = itk.RescaleIntensityImageFilter[FixedImageType,
                                                    OutputImageType].New(
    Input=difference,
    OutputMinimum=itk.NumericTraits[OutputPixelType].min(),
    OutputMaximum=itk.NumericTraits[OutputPixelType].max())


# differential image befor registration
resampler.SetDefaultPixelValue(1)
writer.SetInput(intensityRescaler.GetOutput())
writer.SetFileName(differenceImageAfterFile)
writer.Update()


#differential image after registration
resampler.SetTransform(identityTransform)
writer.SetFileName(differenceImageBeforeFile)
writer.Update()


