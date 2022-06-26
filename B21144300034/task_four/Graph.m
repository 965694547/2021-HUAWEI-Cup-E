 %% 有向环图传播网络
 clc
 clear
 output=zeros(18,4);
 for  class = 1:6
     % 浓度变化矩阵
     [num_h,txt_h,raw_h]=xlsread("环境变化.xlsx");
     [num_a,txt_a,raw_a]=xlsread("A_浓度变化.xlsx");
     [num_a1,txt_a1,raw_a1]=xlsread("A1_浓度变化.xlsx");
     [num_a2,txt_a2,raw_a2]=xlsread("A2_浓度变化.xlsx");
     [num_a3,txt_a3,raw_a3]=xlsread("A3_浓度变化.xlsx");
     % 位置矩阵（0，1，2，3）
     location = [0,0;-14.4846, -1.9699;-6.6716, 7.5953;-3.3543, -5.0138];
     % 数据处理
     MAX = [max(num_h(:,1:3)),1,1];%不处理风速风向
     MIN = [min(num_h(:,1:3)),0,0];
     num_h = (num_h-MIN)./(MAX-MIN);
     wind = num_h(:,4:5);
     wind(:,1) =  num_h(:,4)*3.6;%风速单位化为km/h
     % 方向处理（为方便计算，规定自正西方向至监测点时角度为0）
     wind(:,2)=abs(wind(:,2)-ones(size(wind,1),1)*270);
     wind(:,2) = wind(:,2)/180*pi;
     % 环境影响因子（0-1，0-2，0-3，1-2，1-3，2-3）
     loc = zeros(6,2);%位置向量
     loc(1,1)=location(2,1)-location(1,1);loc(1,2)=location(2,2)-location(1,2);
     loc(2,1)=location(3,1)-location(1,1);loc(2,2)=location(3,2)-location(1,2);
     loc(3,1)=location(4,1)-location(1,1);loc(3,2)=location(4,2)-location(1,2);
     loc(4,1)=location(3,1)-location(2,1);loc(4,2)=location(3,2)-location(2,2);
     loc(5,1)=location(4,1)-location(2,1);loc(5,2)=location(4,2)-location(2,2);
     loc(6,1)=location(4,1)-location(3,1);loc(6,2)=location(4,2)-location(3,2);
     loc1 = zeros(6,2);%位置向量归一
     for i = 1:size(loc,1)
         loc1(i,1)= loc(i,1)/sqrt(loc(i,1)^2+loc(i,2)^2);
         loc1(i,2)= loc(i,2)/sqrt(loc(i,1)^2+loc(i,2)^2);
     end
     w = zeros(1,6);%传递系数
     w_k = ones(1,6);%传递参数
     w_out = ones(1,4);%传递参数
     loss_c = zeros(1,4);%浓度的误差
     loss_k = zeros(1,6);%每个参数的损失
     % 位置浓度
     for time = 1:size(num_h,1)-1-3
         concentration = [num_a(time,class),num_a1(time,class),num_a2(time,class),num_a3(time,class)];%浓度
         env = num_h(time,:);
         for i = 1:size(w,2)
             w(1,i) = wind(time,1)*(cos(wind(time,2))* loc1(i,1)+ sin(wind(time,2))* loc1(i,2)+eps)/sqrt(loc(i,1)^2+loc(i,2)^2);%风速/风向·位置向量
         end
         if(any(isnan(w)))%只要有一条通道为NaN,就跳过
             continue;
         end
         %参数更新
         w_k = tanh(w_k); %激活函数
         %w_k = max(w_k,0); %激活函数
         concentration(1,1)=concentration(1,1)+w(1,1)*w_k(1,1)+w(1,2)*w_k(1,2)+w(1,3)*w_k(1,3);
         concentration(1,2)=concentration(1,2)-w(1,1)*w_k(1,1)+w(1,4)*w_k(1,4)+w(1,5)*w_k(1,5);
         concentration(1,3)=concentration(1,3)-w(1,2)*w_k(1,2)-w(1,4)*w_k(1,4)+w(1,6)*w_k(1,6);
         concentration(1,4)=concentration(1,4)-w(1,3)*w_k(1,3)-w(1,5)*w_k(1,5)-w(1,6)*w_k(1,6);
         %计算损失
         if(~isnan(num_a(time+1,class))&&~isnan(num_a(time,class)))
             loss_c(1,1) =-(num_a(time+1,class) - concentration(1,1));%浓度的误差
             loss_all(1,time)=num_a(time+1,class);
             loss_all(2,time)=concentration(1,1);
         end
         if(~isnan(num_a1(time+1,class))&&~isnan(num_a1(time,class)))
             loss_c(1,2) =-(num_a1(time+1,class) - concentration(1,2));%浓度的误差
             loss_all(3,time)=num_a1(time+1,class);
             loss_all(4,time)=concentration(1,2);
         end
         if(~isnan(num_a2(time+1,class))&&~isnan(num_a2(time,class)))
             loss_c(1,3) =-(num_a2(time+1,class) - concentration(1,3));%浓度的误差
             loss_all(5,time)=num_a2(time+1,class);
             loss_all(6,time)=concentration(1,3);
         end
         if(~isnan(num_a3(time+1,class))&&~isnan(num_a3(time,class)))
             loss_c(1,4) =-(num_a3(time+1,class) - concentration(1,4));%浓度的误差
             loss_all(7,time)=num_a3(time+1,class);
             loss_all(8,time)=concentration(1,4);
         end
         disp(sum(abs(loss_c)));
         if (isnan(sum(loss_c)))
             break
         end
         %梯度回传
         for i = size(loss_c,2)
             loss_c(1,i) = 1-tanh(loss_c(1,i))^2;
         end
         w_k(1,1) = w_k(1,1)-loss_c(1,1)/(w(1,1)*w_k(1,1)+w(1,2)*w_k(1,2)+w(1,3)*w_k(1,3))*(w(1,1)*w_k(1,1)) +  loss_c(1,2)/(-w(1,1)*w_k(1,1)+w(1,4)*w_k(1,4)+w(1,5)*w_k(1,5))*(w(1,1)*w_k(1,1));
         w_k(1,2) = w_k(1,2)-loss_c(1,1)/(w(1,1)*w_k(1,1)+w(1,2)*w_k(1,2)+w(1,3)*w_k(1,3))*(w(1,2)*w_k(1,2)) +  loss_c(1,3)/(-w(1,2)*w_k(1,2)-w(1,4)*w_k(1,4)+w(1,6)*w_k(1,6))*(w(1,2)*w_k(1,2));
         w_k(1,3) = w_k(1,3)-loss_c(1,1)/(w(1,1)*w_k(1,1)+w(1,2)*w_k(1,2)+w(1,3)*w_k(1,3))*(w(1,3)*w_k(1,3)) +  loss_c(1,4)/(-w(1,3)*w_k(1,3)-w(1,5)*w_k(1,5)-w(1,6)*w_k(1,6))*(w(1,3)*w_k(1,3));
         w_k(1,4) = w_k(1,4)-loss_c(1,2)/(-w(1,1)*w_k(1,1)+w(1,4)*w_k(1,4)+w(1,5)*w_k(1,5))*(w(1,4)*w_k(1,4)) +  loss_c(1,3)/(-w(1,2)*w_k(1,2)-w(1,4)*w_k(1,4)+w(1,6)*w_k(1,6))*(w(1,4)*w_k(1,4));
         w_k(1,5) = w_k(1,5)-loss_c(1,2)/(-w(1,1)*w_k(1,1)+w(1,4)*w_k(1,4)+w(1,5)*w_k(1,5))*(w(1,5)*w_k(1,5)) +  loss_c(1,4)/(-w(1,3)*w_k(1,3)-w(1,5)*w_k(1,5)-w(1,6)*w_k(1,6))*(w(1,5)*w_k(1,5));
         w_k(1,6) = w_k(1,6)-loss_c(1,3)/(-w(1,2)*w_k(1,2)-w(1,4)*w_k(1,4)+w(1,6)*w_k(1,6))*(w(1,6)*w_k(1,6)) + loss_c(1,4)/(-w(1,3)*w_k(1,3)-w(1,5)*w_k(1,5)-w(1,6)*w_k(1,6))*(w(1,6)*w_k(1,6)) ;
     end
     figure
     subplot(2,2,1)
     draw= loss_all(1,:);
     plot(draw');
     title('A预测图','FontSize',20);
     xlabel('时间/天','FontSize',20);
     xlabel('浓度','FontSize',20);
     subplot(2,2,2)
     draw= loss_all(2,:);
     plot(draw','Color',[0 0 1]);
     title('A真实图','FontSize',20);
     xlabel('时间/天','FontSize',20);
     xlabel('浓度','FontSize',20);
     subplot(2,2,3)
     draw= loss_all(3,:);
     plot(draw');
     title('A1预测图','FontSize',20);
     xlabel('时间/天','FontSize',20);
     xlabel('浓度','FontSize',20);
     subplot(2,2,4)
     draw= loss_all(4,:);
     plot(draw','Color',[0 0 1]);
     title('A1真实图','FontSize',20);
     xlabel('时间/天','FontSize',20);
     xlabel('浓度','FontSize',20);
     time = 819;
     concentration = [num_a(time,class),num_a1(time,class),num_a2(time,class),num_a3(time,class)];
     for j = 1:3
         for i = 1:size(w,2)
             w(1,i) = wind(time,1)*(cos(wind(time,2))* loc1(i,1)+ sin(wind(time,2))* loc1(i,2)+eps)/sqrt(loc(i,1)^2+loc(i,2)^2);%风速/风向·位置向量
         end
         time = time+1;
         w_k = tanh(w_k); %激活函数
         concentration(1,1)=concentration(1,1)+w(1,1)*w_k(1,1)+w(1,2)*w_k(1,2)+w(1,3)*w_k(1,3);
         concentration(1,2)=concentration(1,2)-w(1,1)*w_k(1,1)+w(1,4)*w_k(1,4)+w(1,5)*w_k(1,5);
         concentration(1,3)=concentration(1,3)-w(1,2)*w_k(1,2)-w(1,4)*w_k(1,4)+w(1,6)*w_k(1,6);
         concentration(1,4)=concentration(1,4)-w(1,3)*w_k(1,3)-w(1,5)*w_k(1,5)-w(1,6)*w_k(1,6);
         output(j+3* class-3,:)=concentration(1,:);
     end
 end
 out = zeros(12,6);
 for i =1:4
     for j = 1:6 
         temp1 = j*3-3+1;
         temp2 = j*3;
         temp3 = i*3-3+1;
         temp4 = i*3;
         out(temp3:temp4,j) = output(temp1:temp2,i);
     end
 end