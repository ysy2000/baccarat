import os
import pandas as pd
import pymysql
from tqdm import tqdm


conn = pymysql.connect(
    user='rovigos',
    passwd='fhqlrhtm1!',
    host='db01.rovigos.com',
    port=13306,
    db='lion_test',
    charset='utf8'
)

class Load:
    def __init__(self):
        try :
            self.conn = pymysql.connect(user='rovigos', passwd='fhqlrhtm1!',  host='db01.rovigos.com', port=13306,db='lion_test', charset='utf8')
            self.sql_inven ="""SELECT * FROM tb_inventory order by month desc"""
            self.sql_sales ="""SELECT * FROM tb_sales"""
            self.sql_product = """SELECT distinct barcode FROM tb_inventory"""
            self.sql_product_sales = """SELECT distinct item_code FROM tb_sales"""
            self.sql_inven_top10p = """
                SELECT * FROM tb_inventory as A
                LEFT JOIN (SELECT distinct barcode FROM tb_inventory ORDER BY RAND() limit 2731) AS B on A.barcode = B.barcode;
            """
            self.cursor = conn.cursor()
            self.columns_inven = None
            self.columns_sales = None
            self.month_file = pd.DataFrame()
            self.columns_desc = {"item_dep" : "모델명, 상품 구분", "item_div01" : "상품 대분류", "item_div02" : "상품 중분류", "item_div03" : "상품 소분류", "rsp": "단위 가격", "qty" : "수량", "net_sales":"실제 판매가"}
        except Exception as e:
            print(e)

    def sqlRun(self, query, data = None):
        try:
            if data == None:
                self.cursor.execute(query)
            else :
                self.cursor.execute(query, data)

            return cursor.fetchall()
        except Exception as e:
            print(e)

    def loadInven(self):
        self.cursor.execute(self.sql_inven)
        self.columns_inven = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(self.cursor.fetchall(), columns=self.columns_inven)

        return df

    def infoInven(self):
        return self.columns_inven

    def loadSales(self):
        self.cursor.execute(self.sql_sales)
        self.columns_sales = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(self.cursor.fetchall(), columns=self.columns_sales)

        return df

    def infoSales(self):
        return self.columns_sales

    def infoCommon(self):
        return self.columns_desc

    def __del__(self):
        self.conn.close()

    def loadProduct(self):
        self.cursor.execute(self.sql_product)
        # self.cursor.execute(self.sql_product_sales)
        self.columns_sales = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(self.cursor.fetchall(), columns=self.columns_sales)

        return df



