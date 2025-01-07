package com.lind.deeplearning;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * @author lind
 * @date 2025/1/7 8:33
 * @since 1.0.0
 */
public class MnistExample {


    public static void main(String[] args) throws Exception {
        //    加载数据集：使用 MnistDataSetIterator 加载 MNIST 数据集。
        //    配置神经网络：
        //    使用 NeuralNetConfiguration.Builder 构建神经网络配置。
        //    添加输入层（DenseLayer）和输出层（OutputLayer）。
        //    创建和初始化模型：使用 MultiLayerNetwork 创建模型并初始化。
        //    训练模型：通过循环调用 fit() 方法训练模型。
        //    加载 MNIST 数据集
        DataSetIterator mnistTrain = new MnistDataSetIterator(128, true, 12345);

        // 配置神经网络
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256).nOut(10).build())
                .build();

        // 创建并初始化网络
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));  // 每100次迭代输出一次分数

        // 训练模型
        for (int i = 0; i < 10; i++) { // 训练10个epoch
            System.out.println("i=" + i);
            model.fit(mnistTrain);
        }

        System.out.println("训练完成！");

        // 加载 MNIST 测试数据集
        DataSetIterator mnistTest = new MnistDataSetIterator(128, false, 12345);

        // 评估模型
        double accuracy = model.evaluate(mnistTest).accuracy();
        System.out.println("模型准确率: " + accuracy);

        // 保存模型到文件
        File modelFile = new File("mnist_model.zip");
        ModelSerializer.writeModel(model, modelFile, true);

    }
}
