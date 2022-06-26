function IAQIn = IAQI(data,type)
    %阶梯数据
    step_co = [0,2,4,14,24,36,48,60];
    step_so2 =[0,50,150,475,500,1600,2100,2620];
    step_no2 =[0,40,80,180,280,565,750,940];
    step_o3 = [0,100,160,215,265,800];
    step_PM10 =[0,50,150,250,350,420,500,600];
    step_PM2_5=[0,35,75,115,150,250,350,500];
    step_IAQI = [0,50,100,150,200,300,400,500];
    %元胞数组整合污染物的阶梯数据
    step = {step_co,step_so2,step_no2,step_o3,step_PM10,step_PM2_5};
    %数值超范围的考虑
    if type == 4 && data >800
            fprintf("臭氧（O3）最大8小时滑动平均浓度值高于800 μg∕m^3 的，不再进行其空气质量分指数计算。");
            IAQIn = 300;
    elseif type ~= 4 && data >step{type}(8)
           fprintf("污染物浓度高于IAQI=500对应限值时，不再进行其空气质量分指数计算。");
           IAQIn = 500;  
    %计算IAQ值
    else
        for i = 1:8
            if data < step{type}(i)    
                IAQIn = (step_IAQI(i)-step_IAQI(i-1))/(step{type}(i)-step{type}(i-1))*(data-step{type}(i-1))+step_IAQI(i-1);
                break
            end
        end
    end
    %四舍五入
    IAQIn = round(IAQIn);
end