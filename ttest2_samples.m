function ttest_res=ttest2_samples(classification_accs)
ttest_res=zeros(size(classification_accs,1),2)
res_index=1;
 for i=1:size(classification_accs,2)
     for j=i+1:size(classification_accs,2)
         fprintf("i=%d, k = %d\n",i,j);
         [h p] = ttest2(classification_accs(i,:),classification_accs(j,:));
         ttest_res(res_index,:)=[h p];
         res_index=res_index+1;
     end
 end
end