package com.lind.deeplearning;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

import static com.lind.deeplearning.MnistUtils.loadGrayImg;

/**
 * 模型测试
 *
 * @author lind
 * @date 2025/1/7 10:24
 * @since 1.0.0
 */
public class MnistTest {

    public static void main(String[] args) throws IOException {
        // 加载已训练的模型
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("E:\\github\\lind-deeplearning4j\\mnist_model.zip"), true);
        // 测试图像路径
        String testImagePath = "d:\\dlj4\\img\\";
        // 假设你有10个测试图像，命名为 0.png 到 9.png，当我从MNIST数据集网站下载9张图片后，这个大模型确实可以给我识别出来
        for (int i = 0; i <= 3; i++) {
            String fileName = testImagePath + i + ".png";
            System.out.println("fileName=" + fileName);
            INDArray testImage = loadGrayImg(fileName);
            INDArray output = model.output(testImage); // 进行预测

            // 获取预测结果
            int predictedClass = Nd4j.argMax(output, 1).getInt(0);
            System.out.println("测试图像 " + i + " 的预测结果: " + predictedClass);
        }
    }
}
