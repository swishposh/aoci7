using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

using Emgu.CV.Features2D;


namespace aoci07
{
    public partial class Form1 : Form
    {
        private Image<Bgr, byte> baseImg; //глобальная переменная
        private Image<Bgr, byte> twistedImg;

        //Image<Bgr, byte> sourceImage;
        //Image<Bgr, byte> secondImage;

        public Form1()
        {
            InitializeComponent();

            imageBox3.Visible = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            var result = openFileDialog.ShowDialog(); // открытие диалога выбора файла
            if (result == DialogResult.OK) // открытие выбранного файла
            {
                string fileName = openFileDialog.FileName;
                baseImg = new Image<Bgr, byte>(fileName);

                imageBox1.Image = baseImg.Resize(640, 480, Inter.Linear);
            }

            result = openFileDialog.ShowDialog(); // открытие диалога выбора файла
            if (result == DialogResult.OK) // открытие выбранного файла
            {
                string fileName = openFileDialog.FileName;
                twistedImg = new Image<Bgr, byte>(fileName);

                imageBox2.Image = twistedImg.Resize(640, 480, Inter.Linear);
            }
            imageBox3.Visible = false;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            GFTTDetector detector = new GFTTDetector(40, 0.01, 5, 3, true);

            MKeyPoint[] GFP1 = detector.Detect(baseImg.Convert<Gray, byte>().Mat);


            //создание массива характерных точек исходного изображения (только позиции)
            PointF[] srcPoints = new PointF[GFP1.Length];
            for (int i = 0; i < GFP1.Length; i++)
                srcPoints[i] = GFP1[i].Point;
            PointF[] destPoints; //массив для хранения позиций точек на изменённом изображении
            byte[] status; //статус точек (найдены/не найдены)
            float[] trackErrors; //ошибки
                                 //вычисление позиций характерных точек на новом изображении методом Лукаса-Канаде
            CvInvoke.CalcOpticalFlowPyrLK(
             baseImg.Convert<Gray, byte>().Mat, //исходное изображение
             twistedImg.Convert<Gray, byte>().Mat,//изменённое изображение
             srcPoints, //массив характерных точек исходного изображения
             new Size(20, 20), //размер окна поиска
             5, //уровни пирамиды
             new MCvTermCriteria(20, 1), //условие остановки вычисления оптического потока
             out destPoints, //позиции характерных точек на новом изображении
             out status, //содержит 1 в элементах, для которых поток был найден
             out trackErrors //содержит ошибки
             );

            //вычисление матрицы гомографии
            Mat homographyMatrix = CvInvoke.FindHomography(destPoints, srcPoints,
            RobustEstimationAlgorithm.LMEDS);
            var destImage = new Image<Bgr, byte>(baseImg.Size);
            CvInvoke.WarpPerspective(twistedImg, destImage, homographyMatrix, destImage.Size);


            //var output1 = baseImg.Clone();

            //foreach (MKeyPoint p in GFP1)
            //{
            //    CvInvoke.Circle(output1, Point.Round(p.Point), 3, new Bgr(Color.Blue).MCvScalar, 2);
            //}
            //imageBox1.Image = output1.Resize(640, 480, Inter.Linear);

            ////var output2 = twistedImg.Clone();

            //foreach (PointF p in destPoints)
            //{
            //    CvInvoke.Circle(destImage, Point.Round(p), 3, new Bgr(Color.Blue).MCvScalar, 2);
            //}
            imageBox2.Image = destImage.Resize(640, 480, Inter.Linear);


        }


        private void button3_Click(object sender, EventArgs e)
        {
            GFTTDetector detector = new GFTTDetector(40, 0.01, 5, 3, true);

            var baseImgGray = baseImg.Convert<Gray, byte>();
            var twistedImgGray = twistedImg.Convert<Gray, byte>();

            //генератор описания ключевых точек
            Brisk descriptor = new Brisk();

            //поскольку в данном случае необходимо посчитать обратное преобразование
            //базой будет являться изменённое изображение
            VectorOfKeyPoint GFP1 = new VectorOfKeyPoint();
            UMat baseDesc = new UMat();
            UMat bimg = twistedImgGray.Mat.GetUMat(AccessType.Read);

            VectorOfKeyPoint GFP2 = new VectorOfKeyPoint();
            UMat twistedDesc = new UMat();
            UMat timg = baseImgGray.Mat.GetUMat(AccessType.Read);

            //получение необработанной информации о характерных точках изображений
            detector.DetectRaw(bimg, GFP1);

            //генерация описания характерных точек изображений
            descriptor.Compute(bimg, GFP1, baseDesc);
            detector.DetectRaw(timg, GFP2);
            descriptor.Compute(timg, GFP2, twistedDesc);

            //класс позволяющий сравнивать описания наборов ключевых точек
            BFMatcher matcher = new BFMatcher(DistanceType.L2);

            //массив для хранения совпадений характерных точек
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            //добавление описания базовых точек
            matcher.Add(baseDesc);
            //сравнение с описанием изменённых
            matcher.KnnMatch(twistedDesc, matches, 2, null);
            //3й параметр - количество ближайших соседей среди которых осуществляется поиск совпадений
            //4й параметр - маска, в данном случае не нужна

            //маска для определения отбрасываемых значений (аномальных и не уникальных)
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            //определение уникальных совпадений
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);

            Mat homography;
            //получение матрицы гомографии
            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(GFP1, GFP2, matches, mask, 2);

            var destImage = new Image<Bgr, byte>(baseImg.Size);
            CvInvoke.WarpPerspective(twistedImg, destImage, homography, destImage.Size);
            twistedImg = destImage;
            imageBox2.Image = destImage.Resize(640, 480, Inter.Linear);


           

        }


        public Image<Bgr, byte> pointComp(Image<Bgr, byte> baseImg, Image<Bgr, byte> twistedImg)
        {
            Image<Gray, byte> baseImgGray = baseImg.Convert<Gray, byte>();
            Image<Gray, byte> twistedImgGray = twistedImg.Convert<Gray, byte>();
            Brisk descriptor = new Brisk();
            GFTTDetector detector = new GFTTDetector(40, 0.01, 5, 3, true);
            VectorOfKeyPoint GFP1 = new VectorOfKeyPoint();
            UMat baseDesc = new UMat();
            UMat bimg = twistedImgGray.Mat.GetUMat(AccessType.Read);
            VectorOfKeyPoint GFP2 = new VectorOfKeyPoint();
            UMat twistedDesc = new UMat();
            UMat timg = baseImgGray.Mat.GetUMat(AccessType.Read);
            detector.DetectRaw(bimg, GFP1);
            descriptor.Compute(bimg, GFP1, baseDesc);
            detector.DetectRaw(timg, GFP2);
            descriptor.Compute(timg, GFP2, twistedDesc);
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            matcher.Add(baseDesc);
            matcher.KnnMatch(twistedDesc, matches, 2, null);
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);
            //int nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(GFP1, GFP1, matches, mask, 1.5, 20);
            Image<Bgr, byte> res = baseImg.CopyBlank();
            Features2DToolbox.DrawMatches(twistedImg, GFP1, baseImg, GFP2, matches, res, new MCvScalar(255, 0, 0), new MCvScalar(255, 0, 0), mask);
            return res;
        }


        
        private void button4_Click(object sender, EventArgs e)
        {
            imageBox3.Visible = true;
            imageBox3.Image = pointComp(baseImg, twistedImg);
        }
    }
}
