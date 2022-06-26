function task1()
    %加载原始数据
    row_data=[0.5,8,12,112,27,11;
            0.5,7,16,92,24,10;
            0.6,7,31,169,37,23;
            0.7,8,30,201,47,33];
    %污染名称
    factors=["CO","SO2","NO2","O3","PM10","PM2.5"];
    %记录所有的IAQ值
    IAQI6 = zeros(4,6);
    %计算所有IAQ值
    for i = 1:4
        for j = 1:6
            IAQI6(i,j)= IAQI(row_data(i,j),j);
        end
    end
    %取最大值
    [IAQ,~]=max(IAQI6,[],2);
    %考虑数值一样时的污染名称
    for i =1:4
        if IAQ(i)>50
            IAQ_name(i) = {factors(IAQI6(i,:)== IAQ(i))};
        else
            IAQ_name(i) = {["none"]};
        end
    end
    x = [25,26,27,28];
    y = IAQI6;
    b= bar(x,y);
    legend(factors,'Location','NorthWest');
    for i = 1:6
        xtips1 = b(i).XEndPoints;
        ytips1 = b(i).YEndPoints;
        labels1 = string(b(i).YData);
        text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
    end
    xlabel('日期');
    ylabel('IAQI'); 
    ylim([0,150]);
    %输出结果
    for i = 1:4
        if IAQ(i)>50 
            fprintf("第%d日的IAQ为%d，首要污染物为",i+24,IAQ(i));
            for j = IAQ_name{i}
                fprintf(j+",\n");
            end
        else
            fprintf("第%d日的IAQ为%d，当天无首要污染物\n",i+24,IAQ(i))
        end
    end
% 第25日的IAQ为60，首要污染物为O3,
% 第26日的IAQ为46，当天无首要污染物
% 第27日的IAQ为108，首要污染物为O3,
% 第28日的IAQ为137，首要污染物为O3,
end

