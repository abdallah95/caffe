
idls = readIDL('lp-annot.idl');
count = 0;
[k, num_images] = size(idls);
for i=1:num_images
    im = idls(1,i);
    copyfile(im.img,'../trainval/images/');
    fid = fopen(['../trainval/annotations/' im.img(6:21) '.txt'],'w');
    for j=1:size(im.bb)
        bb = im.bb(j,:);
        xmin = bb(1); ymin = bb(2);
        xmax = bb(3); ymax = bb(4);
        if abs(ymax-ymin)<=20 || xmin<1 || xmax>640 || ymin<1 || ymax>480 || ymin>ymax || xmin>xmax
            continue;
        end
        count = count + 1;
        fprintf(fid,['%i' repmat(' %i',1,4) '\n'],1,[xmin ymin xmax ymax]);
    end
    fclose(fid);
end
count