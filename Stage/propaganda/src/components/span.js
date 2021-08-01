

class Span{
    begin=0;
    end=0;
    probability=0;
    texteSpan="";
    label=0;
    
    constructor(texteSpan,begin,end,label,probability){
         this.texteSpan=texteSpan
         this.begin=begin;
         this.end=end;
         this.label=label;
         this.probability=probability
         //this.span=span
         /*this.debut=debut;
         this.fin=fin*/
     }
    
     overlap(span1,span2){
        if((span1.begin>=span2.begin && span1.begin<=span2.end) || (span1.end>=span2.begin && span1.end<=span2.end)){
            return true;
        }else if((span2.begin>=span1.begin && span2.end<=span1.end) || (span2.end>=span1.begin && span2.end<=span1.end)){
            return true;
        }
        return false;
     }
    /*getText(Reponse){
        let numero=Reponse.toString().indexOf(this.debut);
        let numero2=Reponse.toString().indexOf(this.fin)+this.fin.length;
        let marque="";
    
        for(let i=numero;i<numero2;i++){
           //console.log(Reponse.toString().charAt(i))
           marque=marque+Reponse.toString().charAt(i).toString();
           }
           return marque;
    }*/
    
    }
