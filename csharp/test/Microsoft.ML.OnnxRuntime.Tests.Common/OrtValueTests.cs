using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Text;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    public class OrtValueTests
    {
        public OrtValueTests()
        {
        }

        [Fact(DisplayName = "PopulateAndReadStringTensor")]
        public void PopulateAndReadStringTensor()
        {
            OrtEnv.Instance();

            var strsRom = new string[] { "HelloR", "OrtR", "WorldR" };
            var strs = new string[] { "Hello", "Ort", "World" };
            var shape = new long[] { 1, 1, 3 };
            var elementsNum = ArrayUtilities.GetSizeForShape(shape);
            Assert.Equal(elementsNum, strs.Length);

            using (var strTensor = OrtValue.CreateTensorWithEmptyStrings(shape))
            {
                Assert.True(strTensor.IsTensor);
                Assert.False(strTensor.IsSparseTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, strTensor.OnnxType);
                using (var typeShape = strTensor.GetTensorTypeAndShape())
                {
                    Assert.True(typeShape.IsString);
                    Assert.Equal(shape.Length, typeShape.GetDimensionsCount());
                    var fetchedShape = typeShape.GetShape();
                    Assert.Equal(shape.Length, fetchedShape.Length);
                    Assert.Equal(shape, fetchedShape);
                    Assert.Equal(elementsNum, typeShape.GetElementCount());
                }

                using (var memInfo = strTensor.GetTensorMemoryInfo())
                {
                    Assert.Equal("Cpu", memInfo.Name);
                    Assert.Equal(OrtMemType.Default, memInfo.GetMemoryType());
                    Assert.Equal(OrtAllocatorType.DeviceAllocator, memInfo.GetAllocatorType());
                }

                // Verify that everything is empty now.
                for (int i = 0; i < elementsNum; ++i)
                {
                    var str = strTensor.GetStringElement(i);
                    Assert.Empty(str);

                    var rom = strTensor.GetStringElementAsMemory(i);
                    Assert.Equal(0, rom.Length);

                    var bytes = strTensor.GetStringElementAsSpan(i);
                    Assert.Equal(0, bytes.Length);
                }

                // Let's populate the tensor with strings.
                for (int i = 0; i < elementsNum; ++i)
                {
                    // First populate via ROM
                    strTensor.FillStringTensorElement(strsRom[i].AsMemory(), i);
                    Assert.Equal(strsRom[i], strTensor.GetStringElement(i));
                    Assert.Equal(strsRom[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.True(Encoding.UTF8.GetBytes(strsRom[i]).AsSpan().SequenceEqual(strTensor.GetStringElementAsSpan(i)));

                    // Fill via Span
                    strTensor.FillStringTensorElement(strs[i].AsSpan(), i);
                    Assert.Equal(strs[i], strTensor.GetStringElement(i));
                    Assert.Equal(strs[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.True(Encoding.UTF8.GetBytes(strs[i]).AsSpan().SequenceEqual(strTensor.GetStringElementAsSpan(i)));
                }
            }
        }

        [Fact(DisplayName = "PopulateAndReadStringTensorViaTensor")]
        public void PopulateAndReadStringTensorViaTensor()
        {
            OrtEnv.Instance();

            var strs = new string[] { "Hello", "Ort", "World" };
            var shape = new int[] { 1, 1, 3 };

            var tensor = new DenseTensor<string>(strs, shape);

            using (var strTensor = OrtValue.CreateStringTensor(tensor))
            {
                Assert.True(strTensor.IsTensor);
                Assert.False(strTensor.IsSparseTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, strTensor.OnnxType);
                using (var typeShape = strTensor.GetTensorTypeAndShape())
                {
                    Assert.True(typeShape.IsString);
                    Assert.Equal(shape.Length, typeShape.GetDimensionsCount());
                    var fetchedShape = typeShape.GetShape();
                    Assert.Equal(shape.Length, fetchedShape.Length);
                    Assert.Equal(strs.Length, typeShape.GetElementCount());
                }

                using (var memInfo = strTensor.GetTensorMemoryInfo())
                {
                    Assert.Equal("Cpu", memInfo.Name);
                    Assert.Equal(OrtMemType.Default, memInfo.GetMemoryType());
                    Assert.Equal(OrtAllocatorType.DeviceAllocator, memInfo.GetAllocatorType());
                }

                for (int i = 0; i < strs.Length; ++i)
                {
                    // Fill via Span
                    Assert.Equal(strs[i], strTensor.GetStringElement(i));
                    Assert.Equal(strs[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.True(Encoding.UTF8.GetBytes(strs[i]).AsSpan().SequenceEqual(strTensor.GetStringElementAsSpan(i)));
                }
            }
        }
    }
}
