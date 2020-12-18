function ttest_res=ttest2_samples(classification_accs)
ttest_res=zeros(10,2)
res_index=1;
fprintf("1=ANN\n");
fprintf("2=DT\n");
fprintf("3=SVM: Linera\n");
fprintf("4=SVM: Polynomial\n");
fprintf("5=SVM: RBF\n");
 for i=1:size(classification_accs,2)
     for j=i+1:size(classification_accs,2)
         fprintf("Comparing between column %d and %d\n",i,j);
         [h p] = ttest2(classification_accs(:,i),classification_accs(:,j));
         ttest_res(res_index,:)=[h p];
         ttest_res(res_index,:)
         res_index=res_index+1;
     end
 end
end