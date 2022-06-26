%% ���ݴ���
clc
clear
[num,txt,raw]=xlsread("����1_2_����A��Сʱ��Ⱦ��Ũ��������ʵ������.xlsx");
array1 = isnan (sum(num(:,:),2));
array2 = (num>0);
array2 = any(~array2,2);
array = bitor(array1,array2);
array = ~[0;array];
raw1 = raw(array,:);%%û�и�����û��NAN��ֵ
xlswrite('����1_2_����A��Сʱ��Ⱦ��Ũ��������ʵ������(��ϴ).xlsx',raw1);
[num1,txt1,raw1]=xlsread('����1_2_����A��Сʱ��Ⱦ��Ũ��������ʵ������(��ϴ).xlsx');
size_n = size(num1);
type = [2,3,5,6,4,1];
IAQ = zeros(1,size_n(1));%�洢��ǰСʱIAQ
data = zeros(6,1);
for i  = 1: size_n(1)
    for j = 1:6
        data(j) = IAQI(num1(i,j),type(j));
        [IAQ(i),~] =  max(data(:));
    end
end
%% �����Ŵ�ģ���˻��㷨��FCM����
IAQ = IAQ';%��һ��Ϊÿ��IAQ �ڶ���Ϊÿ��IAQ�ı任
for i = 1:size(IAQ,1)-1
    IAQ(i,3) = (IAQ(i+1,1)-IAQ(i,1))/IAQ(i,1);
    IAQ(i,2) = (IAQ(i+1,1)-IAQ(i,1));
end
num1 =  num1(:,7:11);%����ÿ��Ļ�������
% �����ݹ�һ������
for i = 1 : size(num1, 2)
    temp = num1(:, i);
    temp = (temp-min(temp))/(max(temp)-min(temp));
    num1(:, i) = temp;
end
m=size(num1,2);% ��������ά��
% ���ĵ㷶Χ[lb;ub]
lb=min(num1);
ub=max(num1);
%% ģ��C��ֵ�������
% ������ָ��Ϊ3������������Ϊ20��Ŀ�꺯������ֹ����Ϊ1e-6
options=[3,20,1e-6];
%�����cn
cn=30;
%% ģ���˻��㷨����
q =0.8;
T0=100; %��ʼ�¶�
Tend=1;%��ֹ�¶�
%% �����Ŵ��㷨����
sizepop=10; %������Ŀ(Numbe of individuals)
MAXGEN=10; %����Ŵ�����(Maximum number of generations)
NVAR=m*cn; %������ά��
PRECI=10; %�����Ķ�����λ��(Precision of variables)
GGAP=0.9; %����(Generation gap)
pc=0.7;
pm=0.01;
trace=zeros(NVAR+1,MAXGEN);
%��������������(Build field descriptor)
FieldD=[rep([PRECI],[1,NVAR]);rep([lb;ub],[1,cn]);rep([1;0;1;1],[1,NVAR])];
Chrom=crtbp(sizepop, NVAR*PRECI); % ������ʼ��Ⱥ
V=bs2rv(Chrom, FieldD);
ObjV=ObjFun(num1,cn,V,options); %�����ʼ��Ⱥ�����Ŀ�꺯��ֵ
T=T0;
while T>Tend
    gen=0;
    while gen<MAXGEN
        %������Ӧ��ֵ
        FitnV=ranking(ObjV);
        SelCh=select('sus', Chrom, FitnV, GGAP); %ѡ��
        SelCh=recombin('xovsp', SelCh,pc); %����
        SelCh=mut(SelCh,pm); %����
        V=bs2rv(SelCh, FieldD);
        ObjVSel=ObjFun(num1,cn,V,options); %�����Ӵ�Ŀ�꺯��ֵ
        [newChrom newObjV]=reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel); %�ز���
        V=bs2rv(newChrom,FieldD);
        %�Ƿ��滻�ɸ���
        for i=1:sizepop
            if ObjV(i)>newObjV(i)
                ObjV(i)=newObjV(i);
                Chrom(i,:)=newChrom(i,:);
            else
                p=rand;
                if p<=exp((newObjV(i)-ObjV(i))/T)
                    ObjV(i)=newObjV(i);
                    Chrom(i,:)=newChrom(i,:);
                end
            end
        end
        gen=gen+1; %������������
        [trace(end,gen),index]=min(ObjV); %�Ŵ��㷨���ܸ���
        trace(1:NVAR,gen)=V(index,:);
        fprintf(1,'%d ',gen);
    end
    T=T*q;
    fprintf(1,'\n�¶�:%1.3f\n',T);
end
[newObjV,center,U]=ObjFun(num1,cn,[trace(1:NVAR,end)]',options); %������ѳ�ʼ�������ĵ�Ŀ�꺯��ֵ
% �鿴������
Jb=newObjV
U=U{1}
center=center{1}
maxU = max(U);
%��¼ÿһ����������
index1 = find(U(1,:) == maxU);
index2 = find(U(2, :) == maxU);
index3 = find(U(3, :) == maxU);
index4 = find(U(4, :) == maxU);
index5 = find(U(5,:) == maxU);
index6 = find(U(6, :) == maxU);
index7 = find(U(7, :) == maxU);
index8 = find(U(8, :) == maxU);
index9 = find(U(9, :) == maxU);
index10 = find(U(10, :) == maxU);

index11 = find(U(11,:) == maxU);
index12 = find(U(12, :) == maxU);
index13 = find(U(13, :) == maxU);
index14 = find(U(14, :) == maxU);
index15 = find(U(15,:) == maxU);
index16 = find(U(16, :) == maxU);
index17 = find(U(17, :) == maxU);
index18 = find(U(18, :) == maxU);
index19 = find(U(19, :) == maxU);
index20 = find(U(20, :) == maxU);

index21 = find(U(21,:) == maxU);
index22 = find(U(22, :) == maxU);
index23 = find(U(23, :) == maxU);
index24 = find(U(24, :) == maxU);
index25 = find(U(25,:) == maxU);
index26 = find(U(26, :) == maxU);
index27 = find(U(27, :) == maxU);
index28 = find(U(28, :) == maxU);
index29 = find(U(29, :) == maxU);
index30 = find(U(30, :) == maxU);
index = {index1,index2,index3,index4,index5,index6,index7,index8,index9,index10,index11,index12,index13,index14,index15,index16,index17,index18,index19,index20,index21,index22,index23,index24,index25,index26,index27,index28,index29,index30};
%����ÿһ����ڻ�����ƽ��Ӱ��
class_sum = zeros(cn,1);
for i=1:size(index,2)
    for j=1:size(index{i},2)
        class_sum(i,1) = IAQ(j,2)+class_sum(i,1);
    end
end
for i=1:size(index,2)
    class_sum(i,1) = class_sum(i,1)/size(index{i},2);
end